/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/graph_utils.h"

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <atomic>
#include <mutex>

#include "graph/ge_context.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_local_context.h"
#include "proto/af_ir.pb.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "mmpa/mmpa_api.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/constant_utils.h"
#include "utils/extern_math_util.h"
#include "ge_dump_graph_whitelist.h"
#include "graph/utils/readable_dump.h"
#include "base/err_msg.h"

namespace af {
enum class DumpGraphLevel {
  kDumpLevel1 = 1,
  kDumpLevel2,
  kDumpLevel3,
  kDumpLevel4,
  kDumpLevelOther,
};

namespace {
const int32_t kBaseOfIntegerValue = 10;
#ifdef FMK_SUPPORT_DUMP
const int32_t kDumpGraphIndexWidth = 8;
#endif

const char_t *const kDumpStrBuild = "Build";
const char_t *const kDumpStrPreRunBegin = "PreRunBegin";
const char_t *const kDumpStrPartition = "partition";
const char_t *const kDumpStrOptimizeSubgraph = "OptimizeSubGraph";
const char_t *const kDumpStrSubgraphFunc = "sub_graph";
const char_t *const kDumpStrAicpu = "Aicpu";
const char_t *const kOriginName4Recover = "_origin_name_4_recover";
const char_t *const kOriginType4Recover = "_origin_type_4_recover";
const char_t *const kLocation4Recover = "_location_4_recover";
const char_t *const kLength4Recover = "_length_4_recover";
const size_t kNameMax = 255U;
const int32_t kCopyGraphMaxRecursionDepth = 10;
const int32_t kNameWidth = 5;
const uint32_t kSubgraphIndexOfPartitionedCall = 0U;
const std::set<std::string> kMergeInputSkipTypes{STREAMACTIVE, STREAMSWITCH, CONSTANT, CONSTANTOP};
constexpr int32_t kInvalidStream = -1;
constexpr size_t kNoOpOptimizeThreshold = 1000UL;

graphStatus ReLinkInputDataEdge(const NodePtr &input_node, const NodePtr &target_node) {
  GE_ASSERT_TRUE(input_node->GetType() == DATA, "Input node: %s should be Data", input_node->GetNamePtr());
  int32_t index = -1;
  (void)AttrUtils::GetInt(input_node->GetOpDesc(), ATTR_NAME_INDEX, index);
  GE_ASSERT_TRUE(index >= 0, "Attr index[%d] of node: %s is invalid, index should not be less than 0", index,
                 input_node->GetNamePtr());
  GE_ASSERT_TRUE(index < static_cast<int32_t>(target_node->GetAllInDataAnchorsSize()),
                 "Attr index[%d] of node: %s cannot be larger than input num: %u of target node: %s", index,
                 input_node->GetNamePtr(), target_node->GetAllInDataAnchorsSize(), target_node->GetNamePtr());
  GELOGD("Begin to handle subgraph input node:%s with index:%d.", input_node->GetName().c_str(), index);
  // get node's in data anchor and peer out anchor
  auto node_in_anchor = target_node->GetInDataAnchor(index);
  GE_ASSERT_NOTNULL(node_in_anchor);
  auto src_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_out_anchor);
  auto data_out_anchor = input_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(data_out_anchor);
  GE_ASSERT_NOTNULL(src_out_anchor->GetOwnerNode());
  GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(src_out_anchor, node_in_anchor), "Remove edge from %s to %s failed.",
                    src_out_anchor->GetOwnerNode()->GetNamePtr(), target_node->GetNamePtr());
  auto node_in_control_anchor = target_node->GetInControlAnchor();
  GE_ASSERT_NOTNULL(node_in_control_anchor);
  for (const auto &peer_in_anchor : data_out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(peer_in_anchor->GetOwnerNode());
    GE_ASSERT_SUCCESS(GraphUtils::ReplaceEdgeSrc(data_out_anchor, peer_in_anchor, src_out_anchor),
                      "Replace src: %s from dst: %s to src: %s failed", input_node->GetNamePtr(),
                      peer_in_anchor->GetOwnerNode()->GetNamePtr(), src_out_anchor->GetOwnerNode()->GetNamePtr());
    // add control edge
    for (const auto &out_anchor : node_in_control_anchor->GetPeerOutControlAnchors()) {
      const auto peer_in_anchor_node = peer_in_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(peer_in_anchor_node);
      GE_ASSERT_NOTNULL(out_anchor->GetOwnerNode());
      GE_ASSERT_SUCCESS(GraphUtils::AddEdge(out_anchor, peer_in_anchor_node->GetInControlAnchor()),
                        "Add control edge from %s to %s failed.", out_anchor->GetOwnerNode()->GetNamePtr(),
                        peer_in_anchor_node->GetNamePtr());
    }
  }
  return SUCCESS;
}

graphStatus RelinkOutputNodeEdge(const NodePtr &out_node, const int32_t out_index, const NodePtr &target_node,
                                 const size_t target_index) {
  // 处理输出算子的连边关系
  GE_ASSERT_TRUE(target_index < static_cast<size_t>(target_node->GetAllOutDataAnchorsSize()),
                 "Attr index[%d] of node: %s cannot be larger than output num: %u of target node: %s", target_index,
                 out_node->GetNamePtr(), target_node->GetAllOutDataAnchorsSize(), target_node->GetNamePtr());
  auto node_out_anchor = target_node->GetOutDataAnchor(target_index);
  GE_ASSERT_NOTNULL(node_out_anchor, "Get index: %zu of node: %s failed", target_index, target_node->GetNamePtr());
  GE_ASSERT_NOTNULL(out_node);
  GELOGD("Begin to handle subgraph output node:%s output:%d with index:%d of node: %s.", out_node->GetNamePtr(),
         out_index, target_index, target_node->GetNamePtr());
  auto src_out_anchor = out_node->GetOutDataAnchor(out_index);
  GE_CHECK_NOTNULL(src_out_anchor);
  for (const auto &dst_in_anchor : node_out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_NOTNULL(dst_in_anchor->GetOwnerNode());
    GE_ASSERT_SUCCESS(GraphUtils::ReplaceEdgeSrc(node_out_anchor, dst_in_anchor, src_out_anchor),
                      "Replace src: %s from dst: %s to src: %s failed", target_node->GetNamePtr(),
                      dst_in_anchor->GetOwnerNode()->GetNamePtr(), out_node->GetNamePtr());
  }
  auto node_out_control_anchor = target_node->GetOutControlAnchor();
  GE_ASSERT_NOTNULL(node_out_control_anchor);
  for (const auto &peer_in_control_anchor : node_out_control_anchor->GetPeerInControlAnchors()) {
    GE_ASSERT_NOTNULL(peer_in_control_anchor->GetOwnerNode());
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(out_node->GetOutControlAnchor(), peer_in_control_anchor),
                      "Add control edge from %s to %s failed.", out_node->GetNamePtr(),
                      peer_in_control_anchor->GetOwnerNode()->GetNamePtr());
  }
  return SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::GetIndependentCompileGraphs(
    const ComputeGraphPtr &compute_graph, std::vector<ComputeGraphPtr> &independent_compile_subgraphs) {
  bool is_pipeline_partitioned = false;
  (void)AttrUtils::GetBool(*compute_graph, ATTR_NAME_PIPELINE_PARTITIONED, is_pipeline_partitioned);
  if (is_pipeline_partitioned) {
    for (const auto &node : compute_graph->GetDirectNode()) {
      if (node->GetType() == PARTITIONEDCALL) {
        auto sub_graph = NodeUtils::GetSubgraph(*node, kSubgraphIndexOfPartitionedCall);
        GE_CHECK_NOTNULL(sub_graph);
        independent_compile_subgraphs.emplace_back(sub_graph);
      }
    }
    return af::GRAPH_SUCCESS;
  }
  independent_compile_subgraphs.emplace_back(compute_graph);
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "addedge failed because param src is nullptr or run LinkTo failed.");
  GELOGE(af::GRAPH_FAILED, "[Add][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const AnchorPtr &src,
                                                                               const AnchorPtr &dst) {
  const OutDataAnchorPtr src_data = Anchor::DynamicAnchorCast<OutDataAnchor>(src);
  const InDataAnchorPtr dst_data = Anchor::DynamicAnchorCast<InDataAnchor>(dst);
  const OutControlAnchorPtr src_control = Anchor::DynamicAnchorCast<OutControlAnchor>(src);
  const InControlAnchorPtr dst_control = Anchor::DynamicAnchorCast<InControlAnchor>(dst);
  if ((src_data != nullptr) && (dst_data != nullptr) && (src_data->LinkTo(dst_data) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  if ((src_data != nullptr) && (dst_control != nullptr) && (src_data->LinkTo(dst_control) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_control != nullptr) && (src_control->LinkTo(dst_control) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_data != nullptr) && (src_control->LinkTo(dst_data) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "AddEdge failed because src or dst is nullptr or run LinkTo failed.");
  GELOGE(af::GRAPH_FAILED, "[Add][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutControlAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(af::GRAPH_FAILED, "[Add][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(af::GRAPH_FAILED, "[Add][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(af::GRAPH_FAILED, "[Remove][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const AnchorPtr &src,
                                                                                  const AnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(af::GRAPH_FAILED, "[Remove][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutControlAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  REPORT_INNER_ERR_MSG("E18888", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(af::GRAPH_FAILED, "[Remove][Edge] Failed.");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  GELOGE(af::GRAPH_FAILED, "[Remove][Edge] Failed, because src is nullptr or Unlink failed");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceEdgeSrc(const OutDataAnchorPtr &src,
                                                                                      const InDataAnchorPtr &dst,
                                                                                      const OutDataAnchorPtr &new_src) {
  if ((RemoveEdge(src, dst) == af::GRAPH_SUCCESS) && (AddEdge(new_src, dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  GELOGE(af::GRAPH_FAILED, "[Replace][EdgeSrc] Failed, because RemoveEdge(src, dst) or AddEdge(new_src, dst) failed");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceEdgeSrc(
    const OutControlAnchorPtr &src, const InControlAnchorPtr &dst, const OutControlAnchorPtr &new_src) {
  if ((RemoveEdge(src, dst) == af::GRAPH_SUCCESS) && (AddEdge(new_src, dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  GELOGE(af::GRAPH_FAILED, "[Replace][EdgeSrc] Failed, because RemoveEdge(src, dst) or AddEdge(new_src, dst) failed");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceEdgeDst(const OutDataAnchorPtr &src,
                                                                                      const InDataAnchorPtr &dst,
                                                                                      const InDataAnchorPtr &new_dst) {
  if ((RemoveEdge(src, dst) == af::GRAPH_SUCCESS) && (AddEdge(src, new_dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  GELOGE(af::GRAPH_FAILED, "[Replace][EdgeDst] Failed, because RemoveEdge(src, dst) or AddEdge(src, new_dst) failed");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceEdgeDst(
    const OutControlAnchorPtr &src, const InControlAnchorPtr &dst, const InControlAnchorPtr &new_dst) {
  if ((RemoveEdge(src, dst) == af::GRAPH_SUCCESS) && (AddEdge(src, new_dst) == af::GRAPH_SUCCESS)) {
    return af::GRAPH_SUCCESS;
  }
  GELOGE(af::GRAPH_FAILED, "[Replace][EdgeDst] Failed, because RemoveEdge(src, dst) or AddEdge(src, new_dst) failed");
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertNodeBetweenDataAnchors(
    const OutDataAnchorPtr &src, const InDataAnchorPtr &dst, const NodePtr &new_node) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(new_node);

  const InDataAnchorPtr node_in_anchor = new_node->GetInDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_in_anchor != nullptr, af::GRAPH_FAILED,
                         "[Invoke][GetInDataAnchor] this node has no inDataAnchor");
  const OutDataAnchorPtr node_out_anchor = new_node->GetOutDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_out_anchor != nullptr, af::GRAPH_FAILED,
                         "[Invoke][GetOutDataAnchor] this node has no outDataAnchor");
  GE_CHK_STATUS_RET(src->Insert(dst, node_in_anchor, node_out_anchor), "[Replace][Peer] Failed, src: %s, dst: %s",
                    src->GetOwnerNode()->GetNamePtr(), dst->GetOwnerNode()->GetNamePtr());
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveSubgraphRecursively(const ComputeGraphPtr &compute_graph, const NodePtr &remove_node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(remove_node);
  GE_CHECK_NOTNULL(remove_node->GetOpDesc());
  if (remove_node->GetOwnerComputeGraph() == nullptr) {
    GELOGW("Node %s has not been set owner graph.", remove_node->GetName().c_str());
    return af::GRAPH_SUCCESS;
  }
  if ((remove_node->GetOwnerComputeGraph() != compute_graph) &&
      (std::find(compute_graph->impl_->nodes_.begin(), compute_graph->impl_->nodes_.end(), remove_node) ==
       compute_graph->impl_->nodes_.end())) {
    GELOGW("Can not find node %s in graph %s.", remove_node->GetName().c_str(), compute_graph->GetName().c_str());
    return af::GRAPH_FAILED;
  }
  if (remove_node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
    GELOGD("Node %s has no subgraph.", remove_node->GetName().c_str());
    return af::GRAPH_SUCCESS;
  }
  // Find all subgraph of this node
  const auto &root_graph = GraphUtils::FindRootGraph(compute_graph);
  std::vector<ComputeGraphPtr> subgraphs;
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;
  NodePtr remove_node_new = remove_node;
  candidates.emplace_back(remove_node_new);
  while (!candidates.empty()) {
    const NodePtr node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    const OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
      auto subgraph = root_graph->GetSubgraph(*name_iter);
      if ((subgraph != nullptr) && (subgraph->impl_ != nullptr)) {
        subgraphs.emplace_back(subgraph);
        (void)candidates.insert(candidates.begin(), subgraph->impl_->nodes_.begin(), subgraph->impl_->nodes_.end());
      }
    }
  }
  // Remove all subgraph
  for (const auto &remove_graph : subgraphs) {
    if (root_graph->RemoveSubGraph(remove_graph) != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "RemoveSubGraph failed, sub graph name is %s, compute graph is %s.",
                           remove_node->GetName().c_str(), compute_graph->GetName().c_str());
      GELOGE(af::GRAPH_FAILED, "[Remove][SubGraph] failed, sub graph name is %s, compute graph is %s.",
             remove_node->GetName().c_str(), compute_graph->GetName().c_str());
      return af::GRAPH_FAILED;
    }
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveNodesByTypeWithoutRelink(const ComputeGraphPtr &compute_graph, const std::string &node_type) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(compute_graph->impl_);
  GELOGI("Start remove %s from graph %s.", node_type.c_str(), compute_graph->GetName().c_str());
  for (auto iter = compute_graph->impl_->input_nodes_.begin(); iter != compute_graph->impl_->input_nodes_.end();) {
    if ((*iter)->GetType() == node_type) {
      iter = compute_graph->impl_->input_nodes_.erase(iter);
    } else {
      iter++;
    }
  }

  for (auto iter = compute_graph->impl_->output_nodes_info_.begin();
       iter != compute_graph->impl_->output_nodes_info_.end();) {
    if (iter->first->GetType() == node_type) {
      iter = compute_graph->impl_->output_nodes_info_.erase(iter);
    } else {
      iter++;
    }
  }

  for (auto iter = compute_graph->impl_->nodes_.begin(); iter != compute_graph->impl_->nodes_.end();) {
    if ((*iter)->GetType() == node_type) {
      if ((node_type != PLACEHOLDER) && (node_type != END)) {
        const auto ret = RemoveSubgraphRecursively(compute_graph, (*iter));
        if (ret != af::GRAPH_SUCCESS) {
          return af::GRAPH_FAILED;
        }
      }
      iter = compute_graph->impl_->nodes_.erase(iter);
      compute_graph->impl_->direct_nodes_size_--;
    } else {
      iter++;
    }
  }
  GELOGI("End remove %s from graph %s.", node_type.c_str(), compute_graph->GetName().c_str());
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(compute_graph->impl_);
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return af::GRAPH_FAILED;
  }

  // If the node save as input node, delete it
  (void)compute_graph->RemoveInputNode(node);

  // If the node save as output node, delete it
  (void)compute_graph->RemoveOutputNode(node);

  // If the node has sub-graphs, delete them
  const auto ret = RemoveSubgraphRecursively(compute_graph, node);
  if (ret != af::GRAPH_SUCCESS) {
    return af::GRAPH_FAILED;
  }

  const auto iter = find(compute_graph->impl_->nodes_.begin(), compute_graph->impl_->nodes_.end(), node);
  if (iter != compute_graph->impl_->nodes_.end()) {
    compute_graph->EraseFromNodeList(iter);
    return af::GRAPH_SUCCESS;
  }
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveNodesWithoutRelink(const ComputeGraphPtr &compute_graph, const std::unordered_set<NodePtr> &nodes) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(compute_graph->impl_);
  for (auto iter = compute_graph->impl_->input_nodes_.begin(); iter != compute_graph->impl_->input_nodes_.end();) {
    if (nodes.count(*iter) > 0U) {
      iter = compute_graph->impl_->input_nodes_.erase(iter);
    } else {
      iter++;
    }
  }

  for (auto iter = compute_graph->impl_->output_nodes_info_.begin();
       iter != compute_graph->impl_->output_nodes_info_.end();) {
    if (nodes.count((*iter).first) > 0U) {
      iter = compute_graph->impl_->output_nodes_info_.erase(iter);
    } else {
      iter++;
    }
  }
  size_t success_removed_nodes_size = 0U;
  for (auto iter = compute_graph->impl_->nodes_.begin(); iter != compute_graph->impl_->nodes_.end();) {
    if (nodes.count(*iter) > 0U) {
      const auto ret = RemoveSubgraphRecursively(compute_graph, (*iter));
      if (ret != af::GRAPH_SUCCESS) {
        return af::GRAPH_FAILED;
      }
      GELOGD("Remove %s from graph %s.", (*iter)->GetName().c_str(), compute_graph->GetName().c_str());
      iter = compute_graph->impl_->nodes_.erase(iter);
      compute_graph->impl_->direct_nodes_size_--;
      success_removed_nodes_size++;
    } else {
      iter++;
    }
  }
  const auto to_be_remove_nodes_size = nodes.size();
  if (success_removed_nodes_size != to_be_remove_nodes_size) {
    GELOGW("Successfully remove %zu nodes but there are %zu nodes to be deleted", success_removed_nodes_size,
           to_be_remove_nodes_size);
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraph &compute_graph,
                                                                                      const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] The node ptr should be not null.");
    return af::GRAPH_FAILED;
  }
  if (compute_graph.impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "The compute graph impl should be not null, check invalid");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] The compute graph impl should be not null.");
    return af::GRAPH_FAILED;
  }
  const auto iter = find(compute_graph.impl_->nodes_.begin(), compute_graph.impl_->nodes_.end(), node);
  if (iter != compute_graph.impl_->nodes_.end()) {
    compute_graph.EraseFromNodeList(iter);
    return af::GRAPH_SUCCESS;
  }
  return af::GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveJustNode(const ComputeGraphPtr compute_graph, const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(node);
  const graphStatus ret =
      ((RemoveJustNode(*compute_graph, node) == af::GRAPH_SUCCESS) ? af::GRAPH_SUCCESS : af::GRAPH_FAILED);
  return ret;
}

void GraphUtils::RecordOriginalNames(const std::vector<af::NodePtr> original_nodes, const af::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
                   return, "[Check][Param] node is null.");
  std::vector<std::string> original_names;
  std::vector<std::string> original_types;
  for (const auto &node_tmp : original_nodes) {
    std::vector<std::string> names_tmp;
    std::vector<std::string> types_tmp;
    const af::OpDescPtr opdesc_tmp = node_tmp->GetOpDesc();
    if (opdesc_tmp == nullptr) {
      GELOGE(af::GRAPH_FAILED, "[Check][Param] Node %s get opdesc is nullptr", node_tmp->GetName().c_str());
      continue;
    }
    auto ret = af::AttrUtils::GetListStr(opdesc_tmp, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp);
    af::AttrUtils::GetListStr(opdesc_tmp, ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, types_tmp);
    if (!ret) {
      GELOGW("[Get][Attr] Get attr _datadump_original_op_names failed");
      continue;
    }
    if (names_tmp.size() != 0UL) {
      (void)original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
    } else {
      original_names.push_back(opdesc_tmp->GetName());
    }
    if (types_tmp.size() != 0UL) {
      (void)original_types.insert(original_types.end(), types_tmp.begin(), types_tmp.end());
    } else {
      original_types.push_back(opdesc_tmp->GetType());
    }
  }
  GE_CHK_BOOL_EXEC(af::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names),
                   REPORT_INNER_ERR_MSG("E18888", "Set original_op_names to node:%s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_names to node:%s fail.", node->GetName().c_str());
  GE_CHK_BOOL_EXEC(af::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, original_types),
                   REPORT_INNER_ERR_MSG("E18888", "Set original_op_types to node:%s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_types to node:%s fail.", node->GetName().c_str());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::RecordOriginalNames(std::vector<std::string> names_tmp,
                                                                                    const af::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
                   return, "[Check][Param] node is null.");
  std::vector<std::string> original_names;
  if (names_tmp.size() != 0UL) {
    (void)original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
  } else {
    const std::string tmp;
    original_names.push_back(tmp);
  }
  GE_CHK_BOOL_EXEC(af::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names),
                   REPORT_INNER_ERR_MSG("E18888", "Set original_op_names to node %s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_names to node %s fail.", node->GetName().c_str());
}

namespace {
#ifdef FMK_SUPPORT_DUMP
void GetDumpGraphPrefix(std::stringstream &stream_file_name) {
  static std::string path_prefix;
  if (path_prefix.empty()) {
    const char_t *npu_collect_path = nullptr;
    MM_SYS_GET_ENV(MM_ENV_NPU_COLLECT_PATH, npu_collect_path);
    if (npu_collect_path != nullptr) {
      const std::string base_path_str(npu_collect_path);
      stream_file_name << base_path_str << "/extra-info/graph/" << mmGetPid() << "_" << GetContext().DeviceId() << "/";
    } else {
      const char_t *dump_graph_path = nullptr;
      MM_SYS_GET_ENV(MM_ENV_DUMP_GRAPH_PATH, dump_graph_path);
      if (dump_graph_path != nullptr) {
        const std::string dump_graph_path_str(dump_graph_path);
        stream_file_name << (dump_graph_path_str.empty() ? "" : dump_graph_path_str + "/");
        stream_file_name << "pid_" << mmGetPid() << "_deviceid_" << GetContext().DeviceId() << "/";
      } else {
        stream_file_name << "./";
        std::string ascend_work_path;
        (void)GetAscendWorkPath(ascend_work_path);
        if (!ascend_work_path.empty()) {
          stream_file_name.str("");
          stream_file_name << (ascend_work_path + "/");
        }
      }
    }
    path_prefix = stream_file_name.str();
  } else {
    stream_file_name << path_prefix;
  }
}

bool SetOptions2GraphInner(const std::map<std::string, std::string> &option, const std::string &attr_name,
                           const af::ComputeGraphPtr &graph) {
  // set graph options
  af::NamedAttrs attr;
  attr.SetName(attr_name);
  for (auto itr_graph = option.begin(); itr_graph != option.end(); itr_graph++) {
    auto const ret = attr.SetAttr(itr_graph->first, GeAttrValue::CreateFrom<std::string>(itr_graph->second));
    if (ret != af::GRAPH_SUCCESS) {
      GELOGE(af::GRAPH_FAILED, "set [%s:] [%s]=[%s] to graph fail.", attr.GetName().c_str(), itr_graph->first.c_str(),
             itr_graph->second.c_str());
      return false;
    }
  }
  auto ret = af::AttrUtils::SetNamedAttrs(graph, attr_name, attr);
  if (!ret) {
    GELOGE(af::GRAPH_FAILED, "set [%s] to graph fail.", attr_name.c_str());
    return false;
  }
  return true;
}
bool SetOptions2Graph(const int64_t dump_level, const af::ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    GELOGE(af::GRAPH_FAILED, "graph is nullptr");
    return false;
  }
  if (dump_level == static_cast<int64_t>(DumpLevel::DUMP_ALL) ||
      dump_level == static_cast<int64_t>(DumpLevel::DUMP_WITH_OUT_DATA)) {
    GEThreadLocalContext &context = GetThreadLocalContext();
    const std::map<std::string, std::string> &tmp_graph_options = context.GetAllGraphOptions();
    const std::map<std::string, std::string> &tmp_session_options = context.GetAllSessionOptions();
    const std::map<std::string, std::string> &tmp_global_options = context.GetAllGlobalOptions();
    if (!SetOptions2GraphInner(tmp_graph_options, "GraphOptions", graph)) {
      return false;
    }
    if (!SetOptions2GraphInner(tmp_global_options, "GlobalOptions", graph)) {
      return false;
    }
    if (!SetOptions2GraphInner(tmp_session_options, "SessionOptions", graph)) {
      return false;
    }
  }
  return true;
}
graphStatus GetDumpRealPath(const int64_t file_index, const std::string &suffix, const std::string &user_graph_name,
                            std::string &real_path_name, const std::string &prefix = "ge_proto_") {
  std::string relative_path;
  if (user_graph_name.empty()) {
    std::stringstream stream_file_name;
    {
      static std::mutex mutex;
      const std::lock_guard<std::mutex> lock(mutex);
      GetDumpGraphPrefix(stream_file_name);
      if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
        if (CreateDir(stream_file_name.str()) != 0) {
          GELOGW("[DumpGraph][CreateDir] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
          stream_file_name.str("");
          stream_file_name << "./";
        }
      }
    }

    stream_file_name << prefix << std::setw(kDumpGraphIndexWidth) << std::setfill('0') << file_index;
    stream_file_name << "_" << GetSanitizedName(suffix) << ".txt";
    relative_path = stream_file_name.str();
  } else {
    const auto sep = user_graph_name.rfind(MMPA_PATH_SEPARATOR_STR);
    if (sep == std::string::npos) {
      (void)relative_path.append("./");
      (void)relative_path.append(user_graph_name);
    } else {
      const std::string file_name = user_graph_name.substr(sep + 1UL, user_graph_name.length());
      std::string path_dir = user_graph_name.substr(0UL, sep + 1UL);
      if ((file_name.length() == 0UL) || (path_dir.length() == 0UL)) {
        GELOGW("[Invalid] path or name is invalid. user_graph_name:%s", user_graph_name.c_str());
        return GRAPH_PARAM_INVALID;
      }

      if ((mmAccess2(path_dir.c_str(), M_F_OK) != EN_OK) && (CreateDir(path_dir) != 0)) {
        GELOGW("[DumpGraph][CreateDir] Create dump graph dir failed, path:%s", path_dir.c_str());
        path_dir = "./";
      }
      (void)relative_path.append(path_dir);
      (void)relative_path.append(file_name);
    }
  }

  char_t real_path[MMPA_MAX_PATH] = {};
  auto const ret = mmRealPath(relative_path.c_str(), &(real_path[0]), MMPA_MAX_PATH);
  if (ret != EN_OK) {
    GELOGD("[Get][RealPath] file does not exist, it will be created. ret:%d", ret);
  }

  real_path_name = real_path;
  GELOGD("Get dump graph real_path_name:%s", real_path_name.c_str());
  return af::GRAPH_SUCCESS;
}

std::string GetDumpFormatStr(DumpFormat dump_format) {
  switch (dump_format) {
    case DumpFormat::GE_PROTO:
      return "ge_proto";
    case DumpFormat::ONNX:
      return "onnx";
    case DumpFormat::READABLE:
      return "readable";
    default:
      return "unknown";
  }
}

bool IsDumpFormatMatch(DumpFormat input_format, const char_t *env_dump_format) {
  if (env_dump_format == nullptr || env_dump_format[0] == '\0') {
    return false;
  }

  std::string env_str(env_dump_format);
  string env_str_trim(StringUtils::Trim(env_str));
  std::string dump_str = GetDumpFormatStr(input_format);

  const auto &format_names = StringUtils::Split(env_str_trim, '|');
  if (std::any_of(format_names.begin(), format_names.end(),
                  [&dump_str](const std::string &format) { return format.find(dump_str) != std::string::npos; })) {
    return true;
  }

  return false;
}

unique_ptr<char[]> ToLowerCase(const char_t *str) {
  auto lower_case_str = ComGraphMakeUnique<char_t[]>(strlen(str) + 1);
  for (int i = 0; str[i] != '\0'; i++) {
    lower_case_str[i] = std::tolower(static_cast<unsigned char>(str[i]));
  }
  lower_case_str[strlen(str)] = str[strlen(str)];

  return lower_case_str;
}

bool NoNeedDumpGraph(int64_t &dump_content_level, DumpFormat format) {
  const char_t *dump_ge_graph = nullptr;
  MM_SYS_GET_ENV(MM_ENV_DUMP_GE_GRAPH, dump_ge_graph);
  if (dump_ge_graph == nullptr) {
    return true;
  }
  const char_t *env_dump_format = nullptr;
  MM_SYS_GET_ENV(MM_ENV_DUMP_GRAPH_FORMAT, env_dump_format);
  if (env_dump_format == nullptr || env_dump_format[0U] == '\0') {
    env_dump_format = "ge_proto|onnx";
  }
  auto lower_case_env_dump_format = ToLowerCase(env_dump_format);
  // 添加防呆，对环境变量统一转换为小写
  if (!IsDumpFormatMatch(format, lower_case_env_dump_format.get())) {
    return true;
  }
  dump_content_level = (dump_ge_graph[0U] != '\0') ? std::strtol(dump_ge_graph, nullptr, kBaseOfIntegerValue)
                                                   : static_cast<int64_t>(DumpLevel::NO_DUMP);
  if ((dump_content_level == static_cast<int64_t>(DumpLevel::NO_DUMP)) ||
      (dump_content_level >= static_cast<int64_t>(DumpLevel::DUMP_LEVEL_END))) {
    GELOGD("Skip dump with DUMP_GE_GRAPH value:%" PRId64 ".", dump_content_level);
    return true;
  }
  return false;
}
#endif

inline graphStatus CheckDumpGraphNum(const int64_t file_index) {
  thread_local int64_t max_dump_file_num = 0;
  if (max_dump_file_num == 0) {
    std::string opt = "0";
    (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_NUM, opt);
    max_dump_file_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if ((max_dump_file_num != 0) && (file_index > max_dump_file_num)) {
    GELOGW("[DumpGraph][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%" PRId64
           ", max_dump_file_num=%" PRId64,
           file_index, max_dump_file_num);
    return GRAPH_PARAM_INVALID;
  }
  return af::GRAPH_SUCCESS;
}
bool IsDumpGraphExcludeSubGraphOnLevel1(const std::string &suffix) {
  if ((suffix.find(kDumpStrPartition) != std::string::npos) ||
      (suffix.find(kDumpStrOptimizeSubgraph) != std::string::npos) ||
      (suffix.find(kDumpStrAicpu) != std::string::npos) || (suffix.find(kDumpStrSubgraphFunc) != std::string::npos)) {
    return false;
  }
  return true;
}

bool IsDumpGraphWithinWhitelistOnLevel2(const std::string &suffix) {
  // 如果是子图则过滤掉
  if (!IsDumpGraphExcludeSubGraphOnLevel1(suffix)) {
    return false;
  }
  for (const auto &full_name : kGeDumpWhitelistFullName) {
    if (suffix.compare(full_name) == 0) {
      return true;
    }
  }
  for (const auto &key_name : kGeDumpWhitelistKeyName) {
    if (suffix.find(key_name) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool IsStrNotNum(const std::string &val) {  // avoid negative number '-'
  for (const auto &ele : val) {
    if (!isdigit(ele)) {
      return true;
    }
  }
  return false;
}

bool IsDumpGraphByKeyName(const std::string &env_val, const std::string &suffix) {
  const auto &key_names = StringUtils::Split(env_val, '|');
  for (const auto &name : key_names) {
    if (suffix.find(name) != std::string::npos) {
      return true;
    }
  }
  return false;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::NoNeedDumpGraphBySuffix(const std::string &suffix) {
  const char_t *dump_level = nullptr;
  MM_SYS_GET_ENV(MM_ENV_DUMP_GRAPH_LEVEL, dump_level);
  if (dump_level == nullptr) {
    return !IsDumpGraphWithinWhitelistOnLevel2(suffix);
  }

  if ((suffix.empty() || (!IsStrNotNum(suffix)))) {
    GELOGW("suffix %s is empty or is number, no need dump", suffix.c_str());
    return true;
  }

  const std::string env_val(dump_level);
  if (IsStrNotNum(env_val)) {
    return !IsDumpGraphByKeyName(env_val, suffix);
  }

  const int64_t dump_graph_level = std::strtol(dump_level, nullptr, kBaseOfIntegerValue);
  if (dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel1)) {
    return !IsDumpGraphExcludeSubGraphOnLevel1(suffix);
  }

  if (dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel2)) {
    return !IsDumpGraphWithinWhitelistOnLevel2(suffix);
  }

  if (dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel3)) {
    return (suffix.compare(kDumpStrBuild) != 0);
  }

  if (dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel4)) {
    return (suffix.compare(kDumpStrPreRunBegin) != 0);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraph(const af::ComputeGraphPtr &graph,
                                                                            const std::string &suffix,
                                                                            const bool is_always_dump,
                                                                            const std::string &user_graph_name) {
#ifdef FMK_SUPPORT_DUMP
  GraphDumperRegistry::GetDumper().Dump(graph, suffix);
  // dump the graph according to different graph level
  int64_t dump_level{0};
  const bool not_dump =
      (NoNeedDumpGraph(dump_level, DumpFormat::GE_PROTO) || GraphUtils::NoNeedDumpGraphBySuffix(suffix)) &&
      (!is_always_dump);
  if (not_dump) {
    return;
  }

  // file name
  std::string real_path;
  if (GenDumpTxtFileName(graph, suffix, user_graph_name, real_path) != af::GRAPH_SUCCESS) {
    return;
  }

  // Create model
  af::Model model("", "");
  if (!SetOptions2Graph(dump_level, graph) && (!is_always_dump)) {
    return;
  }
  model.SetGraph(graph);
  af::proto::ModelDef ge_proto;
  bool is_dump_graph_structure_only = (dump_level != static_cast<int64_t>(DumpLevel::DUMP_ALL)) && (!is_always_dump);
  if (model.Save(ge_proto, is_dump_graph_structure_only) != SUCCESS) {
    return;
  }
  GraphUtils::WriteProtoToTextFile(ge_proto, real_path.c_str());
#else
  (void)graph;
  (void)suffix;
  (void)is_always_dump;
  (void)user_graph_name;
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::DumpGEGraphByPath(
    const af::ComputeGraphPtr &graph, const std::string &file_path, const int64_t dump_level) {
  return DumpGEGraphByPath(graph, file_path, static_cast<DumpLevel>(dump_level));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::DumpGEGraphByPath(
    const af::ComputeGraphPtr &graph, const std::string &file_path, const DumpLevel dump_level) {
  const auto sep = file_path.rfind(MMPA_PATH_SEPARATOR_STR);
  GE_ASSERT_TRUE(sep != std::string::npos, "[CheckParam] Separator is not found in file_path.file_path:%s",
                 file_path.c_str());
  const std::string file_name = file_path.substr(sep + 1UL, file_path.length());
  const std::string path_dir = file_path.substr(0UL, sep + 1UL);
  GE_ASSERT_TRUE((file_name.length() != 0UL) && (path_dir.length() != 0UL),
                 "[Invalid]path or name invalid.file_path:%s", file_path.c_str());

  // Create Model
  af::Model model("", "");
  model.SetGraph(graph);

  // SerializeModel to ModelDef
  af::proto::ModelDef ge_proto;
  if (model.Save(ge_proto, dump_level != DumpLevel::DUMP_ALL) != SUCCESS) {
    return af::GRAPH_FAILED;
  }
  // Write file
  char_t real_path[MMPA_MAX_PATH] = {};
  GE_ASSERT_TRUE(mmRealPath(path_dir.c_str(), &(real_path[0U]), MMPA_MAX_PATH) == EN_OK,
                 "[Get][RealPath]Directory %s does not exist.", path_dir.c_str());
  const std::string path = real_path;
  const std::string real_path_name = path + std::string(MMPA_PATH_SEPARATOR_STR) + file_name;
  GraphUtils::WriteProtoToTextFile(ge_proto, real_path_name.c_str());

  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGrph(const af::ComputeGraphPtr &graph,
                                                                           const std::string &path,
                                                                           const std::string &suffix) {
  // file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump om txt: %" PRId64, file_index);
  if (CheckDumpGraphNum(file_index) != af::GRAPH_SUCCESS) {
    return;
  }

  std::stringstream stream_file_name;
  stream_file_name << path.c_str() << "/ge_proto_" << std::setw(kNameWidth) << std::setfill('0') << file_index;
  stream_file_name << "_graph_" << graph->GetGraphID() << "_" << GetSanitizedName(suffix) << ".txt";
  const std::string proto_file = stream_file_name.str();
  (void)DumpGEGraphByPath(graph, proto_file, DumpLevel::NO_DUMP);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char_t *const file,
                                                                            af::ComputeGraph &compute_graph) {
  af::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  GE_ASSERT_TRUE(ReadProtoFromTextFile(file, &model_def), "[Get][ModelDef] failed from file:%s",
                 (file == nullptr) ? "" : file);
  af::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  GE_ASSERT_SUCCESS(model.Load(model_def), "[Get][Model] failed from ModelDef:%s", file);
  GE_CHK_BOOL_EXEC(model.GetGraph() != nullptr,
                   REPORT_INNER_ERR_MSG("E18888", "Get compute graph is nullptr, model file:%s.", file);
                   return false, "[Get][ComputeGraph] is nullptr");
  compute_graph = *model.GetGraph();
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char_t *const file,
                                                                            af::ComputeGraphPtr &compute_graph) {
  af::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  GE_ASSERT_TRUE(ReadProtoFromTextFile(file, &model_def), "[Get][ModelDef] failed from file:%s",
                 (file == nullptr) ? "" : file);
  af::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  GE_ASSERT_SUCCESS(model.Load(model_def), "[Get][Model] failed from ModelDef:%s", file);
  GE_CHK_BOOL_EXEC(model.GetGraph() != nullptr,
                   REPORT_INNER_ERR_MSG("E18888", "Get compute graph is nullptr, model file:%s.", file);
                   return false, "[Get][ComputeGraph] is nullptr");
  compute_graph = model.GetGraph();
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "ModelDef %s has nullptr node.", file);
      GELOGE(af::GRAPH_FAILED, "[Get][Node]Nullptr node in graph:%s, model:%s", compute_graph->GetName().c_str(), file);
      return false;
    }
    GELOGD("Node %s set owner graph", node->GetName().c_str());
    if (node->SetOwnerComputeGraph(compute_graph) != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "SetOwnerComputeGraph failed for node:%s", node->GetName().c_str());
      GELOGE(af::GRAPH_FAILED, "[Invoke][SetGraph]Node %s set owner graph failed", node->GetName().c_str());
      return false;
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(ConvertFileConstToConst(compute_graph));
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ConvertFileConstToConst(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  std::vector<NodePtr> file_consts;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != FILECONSTANT) {
      continue;
    }
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    std::string file_path;
    if (!AttrUtils::GetStr(op_desc, kLocation4Recover, file_path)) {
      continue;
    }
    GE_ASSERT_TRUE(!file_path.empty());
    GE_ASSERT_GRAPH_SUCCESS(op_desc->DelAttr(kLocation4Recover));
    int64_t attr_length = 0;
    GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, kLength4Recover, attr_length));
    GE_ASSERT_TRUE(attr_length > 0, "attr_length %ld must be greater than 0.", attr_length);
    // Issue #276: upper bound check to prevent OOM from malformed graph files
    constexpr int64_t kMaxFileLength = 2LL * 1024LL * 1024LL * 1024LL;
    GE_ASSERT_TRUE(attr_length <= kMaxFileLength, "attr_length %ld exceeds maximum allowed size %ld.", attr_length,
                   kMaxFileLength);
    GE_ASSERT_GRAPH_SUCCESS(op_desc->DelAttr(kLength4Recover));
    size_t file_length = static_cast<size_t>(attr_length);

    const auto bin_buff = ComGraphMakeUnique<char_t[]>(file_length);
    GE_CHECK_NOTNULL(bin_buff);
    GE_ASSERT_GRAPH_SUCCESS(GetBinFromFile(file_path, bin_buff.get(), file_length));

    const GeTensorPtr &weight = ComGraphMakeShared<GeTensor>(op_desc->GetOutputDesc(0U),
                                                             PtrToPtr<char_t, uint8_t>(bin_buff.get()), file_length);
    GE_CHECK_NOTNULL(weight);

    std::string origin_type;
    if (AttrUtils::GetStr(op_desc, kOriginType4Recover, origin_type) && (kConstOpTypes.count(origin_type) > 0U)) {
      GE_ASSERT_SUCCESS(RecoverConstByWeightFile(op_desc, weight));
      continue;
    }

    const auto const_op = OpDescUtils::CreateConstOp(weight);
    GE_CHECK_NOTNULL(const_op);
    const_op->SetName(op_desc->GetName() + "_" + CONSTANT);
    const_op->SetId(op_desc->GetId());
    const auto const_node = graph->AddNode(const_op);
    GE_CHECK_NOTNULL(const_node);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeAnchors(const_node, node, {}, {0}));
    NodeUtils::UnlinkAll(*node);
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node));
    GELOGD("Convert node: %s from file constant to const by %s success.", node->GetName().c_str(), file_path.c_str());
  }
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RecoverConstByWeightFile(const OpDescPtr &op_desc, const GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(op_desc);
  std::string op_name;
  GE_ASSERT_TRUE(AttrUtils::GetStr(op_desc, kOriginName4Recover, op_name));
  op_desc->SetName(op_name);
  GE_ASSERT_GRAPH_SUCCESS(op_desc->DelAttr(kOriginName4Recover));
  std::string op_type;
  GE_ASSERT_TRUE(AttrUtils::GetStr(op_desc, kOriginType4Recover, op_type));
  op_desc->SetType(op_type);
  GE_ASSERT_GRAPH_SUCCESS(op_desc->DelAttr(kOriginType4Recover));
  GE_ASSERT_TRUE(AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight));
  GELOGD("Recover const node: %s, type: %s.", op_name.c_str(), op_type.c_str());
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::WriteProtoToOStream(const ascend_private::protobuf::Message &proto, std::ostream &o_stream) {
  auto output = ComGraphMakeUnique<google::protobuf::io::OstreamOutputStream>(&o_stream);
  if (output == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create OstreamOutputStream failed.");
    GELOGE(af::GRAPH_FAILED, "[Create][OstreamOutputStream] Output is nullptr");
    return af::GRAPH_FAILED;
  }
  const bool ret = google::protobuf::TextFormat::Print(proto, output.get());
  if (!ret) {
    REPORT_INNER_ERR_MSG("E18888", "write ostream failed.");
    GELOGE(af::GRAPH_FAILED, "[Invoke][Print] Fail to write the ostream");
    return af::GRAPH_FAILED;
  }
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::WriteReadableDumpToOStream(const std::stringstream &readable_ss, std::ostream &o_stream) {
  GE_ASSERT_TRUE(o_stream);
  o_stream << readable_ss.str();
  GE_ASSERT_TRUE(o_stream);

  return SUCCESS;
}

// Printing protocol messages in text format is useful for debugging and human editing of messages.
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::WriteProtoToTextFile(
    const google::protobuf::Message &proto, const char_t *const real_path) {
#ifdef FMK_SUPPORT_DUMP
  const MODE FILE_AUTHORITY = 384U;  // 0600U in octal
  const int32_t fd = mmOpen2(real_path,
                             static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) | static_cast<uint32_t>(M_CREAT) |
                                                  static_cast<uint32_t>(O_TRUNC)),
                             FILE_AUTHORITY);
  if (fd < 0) {
    REPORT_INNER_ERR_MSG("E18888", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(af::GRAPH_FAILED, "[Open][File] failed for %s, reason:%s", real_path, strerror(errno));
    return;
  }

  auto output = ComGraphMakeUnique<google::protobuf::io::FileOutputStream>(fd);
  if (output == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create FileOutputStream failed.");
    GELOGE(af::GRAPH_FAILED, "[Create][FileOutputStream] Output is nullptr");
    if (mmClose(fd) != 0) {
      REPORT_INNER_ERR_MSG("E18888", "close FileOutputStream failed, reason:%s.", strerror(errno));
      GELOGE(af::GRAPH_FAILED, "[Close][FileOutputStream] failed, reason:%s", strerror(errno));
    }
    return;
  }
  const bool ret = google::protobuf::TextFormat::Print(proto, output.get());
  if (!ret) {
    REPORT_INNER_ERR_MSG("E18888", "write file:%s failed.", real_path);
    GELOGE(af::GRAPH_FAILED, "[Invoke][Print] Fail to write the file: %s", real_path);
    GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                     REPORT_INNER_ERR_MSG("E18888", "close FileOutputStream failed, reason:%s.", strerror(errno));
                     return, "[Close][FileOutputStream] failed, reason:%s", strerror(errno));
    return;
  }
  output.reset();
  GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                   REPORT_INNER_ERR_MSG("E18888", "close FileOutputStream failed, reason:%s.", strerror(errno));
                   return, "[Close][FileOutputStream] failed, reason:%s.", strerror(errno));

  FILE *const file = fopen(real_path, "rb");
  if (file == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(af::GRAPH_FAILED, "[Invoke][FOpen] fail to open the file: %s, %s", real_path, strerror(errno));
    return;
  }
  if (fseek(file, 0L, SEEK_END) == 0) {
    const int64_t fileSize = ftell(file);
    thread_local int64_t max_dump_file_size = 0;
    if (max_dump_file_size == 0) {
      std::string opt = "0";
      // Can not check return value
      (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_SIZE, opt);
      max_dump_file_size = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
    }
    if ((max_dump_file_size != 0) && (fileSize != -1) && (fileSize > max_dump_file_size)) {
      GELOGW("[WriteProto][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%" PRId64
             ","
             " max_dump_file_num=%" PRId64,
             fileSize, max_dump_file_size);
      GE_IF_BOOL_EXEC(remove(real_path) != 0, GELOGW("[WriteProto][RemovePath] Remove path %s failed", real_path));
      GE_CHK_BOOL_EXEC(fclose(file) == 0,
                       REPORT_INNER_ERR_MSG("E18888", "close file:%s failed, error:%s", real_path, strerror(errno));
                       return, "[FClose][File] %s failed error:%s", real_path, strerror(errno));
      return;
    }
  }
  GE_CHK_BOOL_EXEC(fclose(file) == 0,
                   REPORT_INNER_ERR_MSG("E18888", "close file:%s failed error:%s", real_path, strerror(errno));
                   return, "[FClose][File] %s failed error:%s", real_path, strerror(errno));
#else
  (void)proto;
  (void)real_path;
  GELOGW("[Write][Proto] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::WriteReadableDumpToTextFile(
    const std::stringstream &readable_ss, const char_t *const real_path) {
#ifdef FMK_SUPPORT_DUMP
  const MODE FILE_AUTHORITY = 384U;  // 0600U in octal
  const int32_t fd = mmOpen2(real_path,
                             static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) | static_cast<uint32_t>(M_CREAT) |
                                                  static_cast<uint32_t>(O_TRUNC)),
                             FILE_AUTHORITY);
  if (fd < 0) {
    REPORT_INNER_ERR_MSG("E18888", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(af::GRAPH_FAILED, "[Open][File] failed for %s, reason:%s", real_path, strerror(errno));
    return;
  }
  const uint32_t len = static_cast<uint32_t>(readable_ss.str().length());
  // Write data to file
  const mmSsize_t mmpa_ret = mmWrite(fd, const_cast<char_t *>(readable_ss.str().c_str()), len);
  if ((mmpa_ret == EN_ERROR) || (mmpa_ret == EN_INVALID_PARAM)) {
    REPORT_INNER_ERR_MSG("E18888", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(af::GRAPH_FAILED, "[Write][File] failed for %s, reason:%s", real_path, strerror(errno));
    return;
  }
  GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                   REPORT_INNER_ERR_MSG("E18888", "close FileOutputStream failed, reason:%s.", strerror(errno));
                   return, "[Close][FileOutputStream] failed, reason:%s.", strerror(errno));
#else
  (void)real_path;
  (void)readable_ss;
  GELOGW("[Write][ReadableDump] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::ReadProtoFromTextFile(
    const char_t *const file, google::protobuf::Message *const proto) {
  if ((file == nullptr) || (proto == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param file or proto is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] incorrect parameter. file path or message is invalid");
    return false;
  }
  std::ifstream fs(file, std::ifstream::in);
  if (!fs.is_open()) {
    REPORT_INNER_ERR_MSG("E18888", "open file:%s failed.", file);
    GELOGE(af::GRAPH_FAILED, "[Invoke][OpenFile]proto file '%s' open fail.", file);
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  const bool ret = google::protobuf::TextFormat::Parse(&input, proto);
  if (!ret) {
    REPORT_INNER_ERR_MSG("E18888", "parse proto from text ret fail, please check your text file '%s'.", file);
    GELOGE(af::GRAPH_FAILED, "[Parse][Proto] from text ret fail, please check your text file '%s'.", file);
  }
  fs.close();
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToOnnx(const af::ComputeGraph &compute_graph,
                                                                                  const std::string &suffix,
                                                                                  bool is_always_dump) {
#ifdef FMK_SUPPORT_DUMP
  // dump the graph according to different graph level
  int64_t dump_content_level{0};
  if (!is_always_dump &&
      (NoNeedDumpGraph(dump_content_level, DumpFormat::ONNX) || GraphUtils::NoNeedDumpGraphBySuffix(suffix))) {
    return;
  }
  return DumpGEGraphToOnnxByContentLevel(compute_graph, suffix, static_cast<DumpLevel>(dump_content_level));
#else
  (void)compute_graph;
  (void)suffix;
  (void)is_always_dump;
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToOnnxByContentLevel(
    const af::ComputeGraph &compute_graph, const std::string &suffix, DumpLevel content_level) {
#ifdef FMK_SUPPORT_DUMP
  // 1.Get af::onnx::ModelProto from ge::Model
  af::Model model("GE", "");
  const auto compute_graph_ptr = ComGraphMakeShared<af::ComputeGraph>(compute_graph);
  model.SetGraph(compute_graph_ptr);
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto, content_level)) {
    GELOGE(af::GRAPH_FAILED, "[Convert][GeModel] DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  std::string real_path;
  if (GenDumpOnnxFileName(compute_graph_ptr, suffix, real_path) != SUCCESS) {
    return;
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.c_str());
#else
  (void)compute_graph;
  (void)suffix;
  (void)content_level;
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToOnnx(const af::ComputeGraph &compute_graph,
                                                                                  const std::string &suffix) {
  return DumpGEGraphToOnnx(compute_graph, suffix, false);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGrphToOnnx(const af::ComputeGraph &compute_graph,
                                                                               const std::string &path,
                                                                               const std::string &suffix) {
  // 1.Get af::onnx::ModelProto from ge::Model
  af::Model model("GE", "");
  const auto compute_graph_ptr = ComGraphMakeShared<af::ComputeGraph>(compute_graph);
  model.SetGraph(compute_graph_ptr);
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGE(af::GRAPH_FAILED, "[Convert][GeModel] DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump ge onnx file: %" PRId64, file_index);
  if (CheckDumpGraphNum(file_index) != af::GRAPH_SUCCESS) {
    return;
  }

  std::stringstream stream_file_name;
  stream_file_name << "ge_onnx_" << std::setw(kNameWidth) << std::setfill('0') << file_index;
  stream_file_name << "_graph_" << compute_graph.GetGraphID();
  stream_file_name << "_" << GetSanitizedName(suffix) << ".pbtxt";
  std::string proto_file = stream_file_name.str();
  if ((proto_file.length()) >= kNameMax) {
    proto_file = proto_file.substr(0U, kNameMax - 7U);
    proto_file = proto_file + ".pbtxt";
    GELOGW("[Check][Param] File name length %zu exceeds max %zu, file:%s", stream_file_name.str().length(), kNameMax,
           stream_file_name.str().c_str());
  }
  const std::string full_proto_file = path + "/" + proto_file;
  const auto real_path = ComGraphMakeUnique<char_t[]>(static_cast<size_t>(MMPA_MAX_PATH));
  if (real_path == nullptr) {
    GELOGE(af::GRAPH_FAILED, "[New][RealPath] failed.");
    return;
  }
  /// Returning nullptr means 3 case as follows:
  /// a.path is PATH_MAX chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  if (mmRealPath(full_proto_file.c_str(), real_path.get(), MMPA_MAX_PATH) != EN_OK) {
    // Case a has been checked above
    GELOGI("File %s does not exist, it will be created, realpath info[%s].", full_proto_file.c_str(), strerror(errno));
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.get());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToReadable(
    const af::ComputeGraphPtr &graph, const std::string &suffix, const bool is_always_dump,
    const std::string &user_graph_name) {
#ifdef FMK_SUPPORT_DUMP
  GraphDumperRegistry::GetDumper().Dump(graph, suffix);
  // dump the graph according to different graph level
  int64_t dump_level{0};
  const bool not_dump =
      (NoNeedDumpGraph(dump_level, DumpFormat::READABLE) || GraphUtils::NoNeedDumpGraphBySuffix(suffix)) &&
      (!is_always_dump);
  if (not_dump) {
    return;
  }

  // file name
  std::string real_path_name;
  GenDumpReadableTxtFileName(graph, suffix, user_graph_name, real_path_name);

  // Create Readable Dump
  std::stringstream readable_ss;
  if (ReadableDump::GenReadableDump(readable_ss, graph) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Dump readable graph %s failed", graph.get()->GetName().c_str());
    GELOGE(af::GRAPH_FAILED, "Dump readable graph %s failed", graph.get()->GetName().c_str());
  }
  WriteReadableDumpToTextFile(readable_ss, real_path_name.c_str());
#else
  (void)graph;
  (void)suffix;
  (void)is_always_dump;
  (void)user_graph_name;
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

namespace {
using InNodesToOut = std::map<NodePtr, std::vector<NodePtr>, NodeCompareKey>;
using OutNodesToIn = InNodesToOut;

inline std::string GetNodeNameByAnchor(const Anchor *const anchor) {
  if (anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param anchor is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Anchor is nullptr");
    return "Null";
  }
  const auto node = anchor->GetOwnerNodeBarePtr();
  return (node == nullptr) ? "Null" : node->GetName();
}

graphStatus ReplaceOutDataAnchor(const OutDataAnchorPtr &new_anchor, const OutDataAnchorPtr &old_anchor,
                                 InNodesToOut *const in_nodes_to_out = nullptr,
                                 OutNodesToIn *const out_nodes_to_in = nullptr) {
  if ((new_anchor == nullptr) || (old_anchor == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param new_anchor or old_anchor is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] new_anchor or old_anchor is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  const auto new_node = new_anchor->GetOwnerNode();
  for (const auto &peer_in_anchor : old_anchor->GetPeerInDataAnchors()) {
    auto ret = peer_in_anchor->Unlink(old_anchor);
    if (ret != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to unlink old anchor link from %s(%d) to %s(%d)",
                           GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
                           GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      GELOGE(af::GRAPH_FAILED, "[Remove][Link] Failed to unlink old anchor link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return af::GRAPH_FAILED;
    }
    ret = peer_in_anchor->LinkFrom(new_anchor);
    if (ret != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "[Create][Link] Failed to relink new anchors from %s(%d) to %s(%d)",
                           GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
                           GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      GELOGE(af::GRAPH_FAILED, "[Create][Link] Failed to relink new anchors from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return af::GRAPH_FAILED;
    }

    if (in_nodes_to_out != nullptr) {
      (void)(*in_nodes_to_out)[new_node].emplace_back(peer_in_anchor->GetOwnerNode());
    }
    if (out_nodes_to_in != nullptr) {
      (void)(*out_nodes_to_in)[peer_in_anchor->GetOwnerNode()].emplace_back(new_node);
    }
  }
  return af::GRAPH_SUCCESS;
}

graphStatus RelinkDataIO(const NodePtr &node, const std::vector<int> &io_map, InNodesToOut &in_nodes_to_out,
                         OutNodesToIn &out_nodes_to_in) {
  GE_CHECK_NOTNULL(node);
  auto in_data_anchors = node->GetAllInDataAnchors();
  auto out_data_anchors = node->GetAllOutDataAnchors();
  const size_t out_data_anchors_size = out_data_anchors.size();
  if (out_data_anchors_size < io_map.size()) {
    REPORT_INNER_ERR_MSG("E18888", "param io_map size:%zu > the actual size:%zu, node:%s type:%s", io_map.size(),
                         out_data_anchors.size(), node->GetName().c_str(), node->GetType().c_str());
    GELOGE(af::GRAPH_FAILED,
           "[Check][Param] The io_map specified for node %s type %s is larger %zu than "
           "the actual size %zu",
           node->GetName().c_str(), node->GetType().c_str(), io_map.size(), out_data_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0U; i < out_data_anchors_size; ++i) {
    const auto out_data_anchor = out_data_anchors.at(i);
    if (out_data_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Failed to relink for node %s type %s, the out data anchor "
                           "at index %zu is null",
                           node->GetName().c_str(), node->GetType().c_str(), i);
      GELOGE(af::GRAPH_FAILED,
             "[Check][Param] Failed to relink for node %s type %s, the out data anchor "
             "at index %zu is null",
             node->GetName().c_str(), node->GetType().c_str(), i);
      return af::GRAPH_FAILED;
    }

    int32_t in_index = -1;
    if (i < io_map.size()) {
      in_index = io_map.at(i);
    }
    if (in_index < 0) {
      out_data_anchor->UnlinkAll();
    } else {
      if (in_index >= static_cast<int32_t>(in_data_anchors.size())) {
        REPORT_INNER_ERR_MSG("E18888",
                             "Failed to relink for node %s type %s, invalid index %d specified for input(%zu)",
                             node->GetName().c_str(), node->GetType().c_str(), in_index, in_data_anchors.size());
        GELOGE(GRAPH_PARAM_INVALID,
               "[Check][Param] Failed to relink for node %s type %s, invalid index %d "
               "specified for input(%zu)",
               node->GetName().c_str(), node->GetType().c_str(), in_index, in_data_anchors.size());
        return GRAPH_PARAM_INVALID;
      }
      const auto in_anchor = in_data_anchors.at(static_cast<size_t>(in_index));
      if (in_anchor == nullptr) {
        GELOGW("[Relink][Check] %d\'th in_data_anchor of node %s type %s is null, ignore it.", in_index,
               node->GetName().c_str(), node->GetType().c_str());
        continue;
      }
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      const auto ret = ReplaceOutDataAnchor(peer_out_anchor, out_data_anchor, &in_nodes_to_out, &out_nodes_to_in);
      if (ret != af::GRAPH_SUCCESS) {
        GELOGE(af::GRAPH_FAILED, "[Replace][OutDataAnchor] Failed to relink node %s type %s for relinking data anchors",
               node->GetName().c_str(), node->GetType().c_str());
        return af::GRAPH_FAILED;
      }
    }
  }

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    in_anchor->UnlinkAll();
  }
  return af::GRAPH_SUCCESS;
}

InNodesToOut GetFullConnectIONodes(const NodePtr &node, std::set<NodePtr> &in_nodes_set,
                                   std::set<NodePtr> &out_nodes_set) {
  InNodesToOut in_nodes_to_out;
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return in_nodes_to_out;
  }
  const auto in_nodes_list = node->GetInNodes();
  const auto out_nodes_list = node->GetOutNodes();
  auto out_nodes = std::vector<NodePtr>(out_nodes_list.begin(), out_nodes_list.end());
  in_nodes_set = std::set<NodePtr>(in_nodes_list.begin(), in_nodes_list.end());
  out_nodes_set = std::set<NodePtr>(out_nodes_list.begin(), out_nodes_list.end());

  for (const auto &in_node : in_nodes_list) {
    (void)in_nodes_to_out.emplace(in_node, out_nodes);
  }
  return in_nodes_to_out;
}

graphStatus RelinkControlNodeIfNeed(const NodePtr &node, const InNodesToOut &in_nodes_to_out,
                                    InNodesToOut &connected_data_in_to_out) {
  GE_CHECK_NOTNULL(node);
  for (const auto &in_node_to_out : in_nodes_to_out) {
    auto &in_node = in_node_to_out.first;
    GE_CHECK_NOTNULL(in_node);
    auto &connected_data_out = connected_data_in_to_out[in_node];
    for (const auto &out_node : in_node_to_out.second) {
      GE_CHECK_NOTNULL(out_node);
      if (std::find(connected_data_out.begin(), connected_data_out.end(), out_node) == connected_data_out.end()) {
        GE_CHECK_NOTNULL(in_node->GetOutControlAnchor());
        if (in_node->GetOutControlAnchor()->IsLinkedWith(out_node->GetInControlAnchor())) {
          continue;
        }
        // Some pass, such as SameTransdataBreadFusionPass will generate a ring, so add a
        // ring breaking operation here, and notice, this is an operation which will be
        // delete later, so do not use this interface to break a ring
        if (in_node == out_node) {
          GELOGW("[Relink][CtrlNode] There is a cycle between %s to %s when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          continue;
        }
        const auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), out_node->GetInControlAnchor());
        if (ret != af::GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E18888", "Add ControlEdge from %s to %s failed, when isolating node %s type %s",
                               in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                               node->GetType().c_str());
          GELOGE(af::GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed, when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          return af::GRAPH_FAILED;
        }
      }
    }
  }
  return af::GRAPH_SUCCESS;
}
template <typename OutDataAnchorVisitor>
graphStatus ReplaceOutDataAnchors(const OutDataAnchorVisitor &new_outs, const OutDataAnchorVisitor &old_outs,
                                  const std::vector<int32_t> &outputs_map) {
  const auto new_out_size = new_outs.size();
  if (new_out_size < outputs_map.size()) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Failed to replace out data anchors, the actual size %zu is less than "
                         "the mapping size %zu",
                         new_out_size, outputs_map.size());
    GELOGE(GRAPH_PARAM_INVALID,
           "[Check][Param] Failed to replace out data anchors, the actual size %zu is less than "
           "the mapping size %zu",
           new_out_size, outputs_map.size());
    return GRAPH_PARAM_INVALID;
  }
  for (size_t i = 0U; i < new_out_size; ++i) {
    auto &new_out_anchor = new_outs.at(i);
    if (new_out_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Failed to replace out data anchors, "
                           "the out data anchor on new node is null, index %zu",
                           i);
      GELOGE(af::GRAPH_FAILED,
             "[Check][Param] Failed to replace out data anchors, "
             "the out data anchor on new node is null, index %zu",
             i);
      return af::GRAPH_FAILED;
    }
    if (i >= outputs_map.size()) {
      continue;
    }
    const auto old_index = outputs_map.at(i);
    if ((old_index < 0) || (static_cast<size_t>(old_index) >= old_outs.size())) {
      continue;
    }

    const OutDataAnchorPtr &old_out_anchor = old_outs.at(static_cast<size_t>(old_index));
    if (old_out_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Failed to replace out data anchors, "
                           "the out data anchor on old node is null, index %d",
                           old_index);
      GELOGE(af::GRAPH_FAILED,
             "[Check][Param] Failed to replace out data anchors, "
             "the out data anchor on old node is null, index %d",
             old_index);
      return af::GRAPH_FAILED;
    }
    const auto ret = ReplaceOutDataAnchor(new_out_anchor, old_out_anchor);
    if (ret != af::GRAPH_SUCCESS) {
      return ret;
    }
  }

  return af::GRAPH_SUCCESS;
}
template <typename InDataAnchorVisitor>
graphStatus DoReplaceInDataAnchors(const InDataAnchorVisitor &new_ins, const InDataAnchorVisitor &old_ins,
                                   const std::vector<int32_t> &inputs_map, bool need_keep_origin = false) {
  const auto new_in_size = new_ins.size();
  if (new_in_size < inputs_map.size()) {
    REPORT_INNER_ERR_MSG("E18888",
                         "Failed to replace in data anchors, "
                         "the actual size %zu is less than the mapping size %zu",
                         new_in_size, inputs_map.size());
    GELOGE(af::GRAPH_FAILED,
           "[Check][Param] Failed to replace in data anchors, "
           "the actual size %zu is less than the mapping size %zu",
           new_in_size, inputs_map.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0U; i < new_in_size; ++i) {
    auto &new_in_anchor = new_ins.at(i);
    if (new_in_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Failed to replace in data anchors, "
                           "the out data anchor on new node is null, index %zu",
                           i);
      GELOGE(af::GRAPH_FAILED,
             "[Check][Param] Failed to replace in data anchors, "
             "the out data anchor on new node is null, index %zu",
             i);
      return af::GRAPH_FAILED;
    }
    if (i >= inputs_map.size()) {
      continue;
    }
    const auto old_index = inputs_map.at(i);
    if ((old_index < 0) || (static_cast<size_t>(old_index) >= old_ins.size())) {
      continue;
    }
    const InDataAnchorPtr &old_in_anchor = old_ins.at(static_cast<size_t>(old_index));
    if (old_in_anchor == nullptr) {
      REPORT_INNER_ERR_MSG("E18888",
                           "Failed to replace in data anchors, "
                           "the out data anchor on old node is null, index %d",
                           old_index);
      GELOGE(af::GRAPH_FAILED,
             "[Check][Param] Failed to replace in data anchors, "
             "the out data anchor on old node is null, index %d",
             old_index);
      return af::GRAPH_FAILED;
    }

    const auto peer_out_anchor = old_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto ret = af::GRAPH_SUCCESS;
    if (!need_keep_origin) {
      ret = peer_out_anchor->Unlink(old_in_anchor);
      if (ret != af::GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
                             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
        GELOGE(af::GRAPH_FAILED, "[Remove][Link] Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
               GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
               GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
        return af::GRAPH_FAILED;
      }
    }
    ret = peer_out_anchor->LinkTo(new_in_anchor);
    if (ret != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to link new anchors, link from %s(%d) to %s(%d)",
                           GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                           GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      GELOGE(af::GRAPH_FAILED, "[Create][Link] Failed to link new anchors, link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return af::GRAPH_FAILED;
    }
  }
  return af::GRAPH_SUCCESS;
}

graphStatus ReplaceControlAnchors(const NodePtr &new_node, const NodePtr &old_node) {
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(new_node->GetInControlAnchor());
  GE_CHECK_NOTNULL(old_node);
  GE_CHECK_NOTNULL(old_node->GetInControlAnchor());
  const auto peer_out_anchors = old_node->GetInControlAnchor()->GetPeerAnchors();
  const auto new_in_control_anchor = new_node->GetInControlAnchor();
  const auto exists_out_anchors = new_in_control_anchor->GetPeerAnchors();
  const auto exists_out_anchors_set = std::set<AnchorPtr>(exists_out_anchors.begin(), exists_out_anchors.end());
  for (const auto &peer_out_anchor : peer_out_anchors) {
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (exists_out_anchors_set.count(peer_out_anchor) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::AddEdge(peer_out_anchor, new_in_control_anchor);
    if (ret != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Add edge from %s to %s failed, ret:%u",
                           peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                           new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
      GELOGE(af::GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
      return af::GRAPH_FAILED;
    }
  }
  const auto old_out_control_anchor = old_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(old_out_control_anchor);
  const auto peer_in_anchors = old_out_control_anchor->GetPeerAnchors();
  const auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  auto exists_in_anchors = new_out_control_anchor->GetPeerAnchors();
  const auto exists_in_anchors_set = std::set<AnchorPtr>(exists_in_anchors.begin(), exists_in_anchors.end());
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (exists_in_anchors_set.count(peer_in_anchor) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::AddEdge(new_out_control_anchor, peer_in_anchor);
    if (ret != af::GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "AddEdge from %s to %s failed, ret:%u",
                           new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                           peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
      GELOGE(af::GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
      return af::GRAPH_FAILED;
    }
  }

  return af::GRAPH_SUCCESS;
}

// check refdata in subgraph is ref from inner data
graphStatus CheckIsRefFromInnerData(const OutDataAnchorPtr &out_data_anchor, NodePtr &inner_data,
                                    bool &is_ref_from_innerdata) {
  is_ref_from_innerdata = false;
  const auto owner_node = out_data_anchor->GetOwnerNode();
  if (owner_node->GetType() != REFDATA) {
    return af::GRAPH_SUCCESS;
  }
  GE_ASSERT_NOTNULL(owner_node->GetOwnerComputeGraph());
  if (owner_node->GetOwnerComputeGraph()->GetParentNode() == nullptr) {
    return af::GRAPH_SUCCESS;
  }

  NodePtr peer_in_ctrl_inner_data = nullptr;
  for (const auto &peer_out_ctrl : owner_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
    const auto peer_in_ctrl_node = peer_out_ctrl->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_in_ctrl_node);
    if (OpTypeUtils::IsSubgraphInnerData(peer_in_ctrl_node->GetOpDesc())) {
      peer_in_ctrl_inner_data = peer_in_ctrl_node;
      break;
    }
  }
  GE_ASSERT_NOTNULL(peer_in_ctrl_inner_data,
                    "Invalid graph. Refdata[%s] in subgraph[%s] should have one control edge from inner data.",
                    owner_node->GetNamePtr(), owner_node->GetOwnerComputeGraph()->GetName().c_str());
  inner_data = peer_in_ctrl_inner_data;
  is_ref_from_innerdata = true;
  return af::GRAPH_SUCCESS;
}

graphStatus CheckIsRefFromRefData(const OutDataAnchorPtr &out_data_anchor, NodePtr &refed_node,
                                  bool &is_ref_from_refdata) {
  is_ref_from_refdata = false;
  const auto owner_node = out_data_anchor->GetOwnerNode();
  const auto out_desc = owner_node->GetOpDesc()->GetOutputDescPtr(static_cast<uint32_t>(out_data_anchor->GetIdx()));
  GE_ASSERT_NOTNULL(out_desc);
  std::string ref_var_src_var_name;
  bool has_ref_attr = af::AttrUtils::GetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
  if (!has_ref_attr) {
    return af::GRAPH_SUCCESS;
  }
  // find src ref_data_node
  const auto &ower_graph = owner_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(ower_graph);
  const auto ref_data_node = ower_graph->FindNode(ref_var_src_var_name);
  if (ref_data_node == nullptr) {
    GELOGW("Can not find refdata named %s. Please check ref relation on graph.", ref_var_src_var_name.c_str());
    return af::GRAPH_SUCCESS;
  }
  if (ref_data_node->GetType() != REFDATA) {
    return af::GRAPH_SUCCESS;
  }
  refed_node = ref_data_node;
  is_ref_from_refdata = true;
  return af::GRAPH_SUCCESS;
}

bool IsNeedOptimizeWithNoOp(const size_t in_size, const size_t out_size) {
  return ((in_size * out_size) > kNoOpOptimizeThreshold) && ((in_size * out_size) > (in_size + out_size));
}

graphStatus RelinkControlNodeWithNoOpOptimize(const NodePtr &node, const std::set<NodePtr> &in_nodes,
                                              const std::set<NodePtr> &out_nodes,
                                              InNodesToOut &connected_data_in_to_out,
                                              InNodesToOut &connected_data_out_to_in) {
  GE_CHECK_NOTNULL(node);
  const auto in_node_size = in_nodes.size();
  const auto out_node_size = out_nodes.size();
  GELOGD("Relink control node with NoOp optimize for [%s][%s], as in_node_size is %zu, out_node_size is %zu",
         node->GetNamePtr(), node->GetTypePtr(), in_node_size, out_node_size);
  std::vector<NodePtr> noop_in_nodes{};
  std::vector<NodePtr> noop_out_nodes{};
  for (const auto &in_node : in_nodes) {
    GE_ASSERT_NOTNULL(in_node);
    const auto &iter = connected_data_in_to_out.find(in_node);
    if ((iter != connected_data_in_to_out.end()) && (iter->second.size() == out_node_size)) {
      GELOGD("The node %s will not add out ctrl to NoOp, as already link to all out nodes", in_node->GetNamePtr());
      continue;
    }
    noop_in_nodes.emplace_back(in_node);
    GELOGD("The node %s will add ctrl to NoOp", in_node->GetNamePtr());
  }
  for (const auto &out_node : out_nodes) {
    GE_ASSERT_NOTNULL(out_node);
    const auto &iter = connected_data_out_to_in.find(out_node);
    if ((iter != connected_data_out_to_in.end()) && (iter->second.size() == in_node_size)) {
      GELOGD("The node %s will not add in ctrl from NoOp, as already link to all in nodes", out_node->GetNamePtr());
      continue;
    }
    noop_out_nodes.emplace_back(out_node);
    GELOGD("The node %s will add in ctrl from NoOp", out_node->GetNamePtr());
  }

  if (noop_in_nodes.empty() || noop_out_nodes.empty()) {
    return af::GRAPH_SUCCESS;
  }

  const auto &graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph);
  const auto &noop = graph->AddNode(OpDescBuilder(node->GetName() + "_noop", NOOP).Build());
  GE_ASSERT_NOTNULL(noop);
  for (const auto &in_node : noop_in_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(in_node->GetOutControlAnchor(), noop->GetInControlAnchor()),
                            "Add ControlEdge from %s to %s failed, when isolating node %s type %s",
                            in_node->GetNamePtr(), noop->GetNamePtr(), node->GetNamePtr(), node->GetTypePtr());
  }

  for (const auto &out_node : noop_out_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(noop->GetOutControlAnchor(), out_node->GetInControlAnchor()),
                            "Add ControlEdge from %s to %s failed, when isolating node %s type %s", noop->GetNamePtr(),
                            out_node->GetNamePtr(), node->GetNamePtr(), node->GetTypePtr());
  }
  return af::GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNode(const NodePtr &node,
                                                                                   const std::vector<int32_t> &io_map) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to isolate node(null)");
    return GRAPH_PARAM_INVALID;
  }

  /// We must get full connections info before re-link data io, because the data
  /// edges may be unlinked when relink data io
  std::set<NodePtr> in_nodes{};
  std::set<NodePtr> out_nodes{};
  const auto in_nodes_to_out = GetFullConnectIONodes(node, in_nodes, out_nodes);

  InNodesToOut data_in_to_out;
  OutNodesToIn data_out_to_in;
  auto ret = RelinkDataIO(node, io_map, data_in_to_out, data_out_to_in);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "[Relink][DataIO] failed, node %s type %s", node->GetName().c_str(),
           node->GetType().c_str());
    return ret;
  }

  if (IsNeedOptimizeWithNoOp(in_nodes.size(), out_nodes.size())) {
    ret = RelinkControlNodeWithNoOpOptimize(node, in_nodes, out_nodes, data_in_to_out, data_out_to_in);
  } else {
    ret = RelinkControlNodeIfNeed(node, in_nodes_to_out, data_in_to_out);
  }
  if (ret != af::GRAPH_SUCCESS) {
    return ret;
  }
  NodeUtils::UnlinkAll(*node);

  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::IsolateNode(const NodePtr &node, const std::initializer_list<int32_t> &io_map) {
  return IsolateNode(node, std::vector<int32_t>(io_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNodeOneIO(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] incorrect parameter. node is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllInDataAnchorsSize() != 1U) {
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllOutDataAnchorsSize() != 1U) {
    return GRAPH_PARAM_INVALID;
  }
  return IsolateNode(node, {0});
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node, const std::vector<int32_t> &inputs_map,
                               const std::vector<int32_t> &outputs_map) {
  if ((new_node == nullptr) || (old_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param new_node or old_node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto ret = ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  if (ret != af::GRAPH_SUCCESS) {
    // The error log was printed in `ReplaceNodeDataAnchors`
    return af::GRAPH_FAILED;
  }
  ret = ReplaceControlAnchors(new_node, old_node);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED,
           "[Replace][ControlAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s",
           old_node->GetName().c_str(), old_node->GetType().c_str(), new_node->GetName().c_str(),
           new_node->GetType().c_str());
    return af::GRAPH_FAILED;
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceNodeAnchors(
    const NodePtr &new_node, const NodePtr &old_node, const std::initializer_list<int32_t> inputs_map,
    const std::initializer_list<int32_t> outputs_map) {
  return ReplaceNodeAnchors(new_node, old_node, std::vector<int32_t>(inputs_map), std::vector<int32_t>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceNodeDataAnchors(
    const NodePtr &new_node, const NodePtr &old_node, const std::initializer_list<int32_t> inputs_map,
    const std::initializer_list<int32_t> outputs_map) {
  return ReplaceNodeDataAnchors(new_node, old_node, std::vector<int32_t>(inputs_map),
                                std::vector<int32_t>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                   const std::vector<int32_t> &inputs_map, const std::vector<int32_t> &outputs_map) {
  if ((new_node == nullptr) || (old_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param new_node or old_node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }

  auto ret = ReplaceOutDataAnchors(new_node->GetAllOutDataAnchors(), old_node->GetAllOutDataAnchors(), outputs_map);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED,
           "[Replace][OutDataAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s",
           old_node->GetName().c_str(), old_node->GetType().c_str(), new_node->GetName().c_str(),
           new_node->GetType().c_str());
    return af::GRAPH_FAILED;
  }
  ret = DoReplaceInDataAnchors(new_node->GetAllInDataAnchors(), old_node->GetAllInDataAnchors(), inputs_map);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED,
           "[Replace][InDataAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s",
           old_node->GetName().c_str(), old_node->GetType().c_str(), new_node->GetName().c_str(),
           new_node->GetType().c_str());
    return af::GRAPH_FAILED;
  }
  return af::GRAPH_SUCCESS;
}

// 检查并插入owner图是否唯一
bool CheckAndInsertGraph(const NodePtr &node, std::set<ComputeGraph *> &owner_graph,
                         ComputeGraph **graph_has_inserted_to_set) {
  auto graph = node->GetOwnerComputeGraphBarePtr();
  if (graph != nullptr) {
    if (owner_graph.empty()) {
      GE_ASSERT_TRUE(owner_graph.insert(graph).second);
      *graph_has_inserted_to_set = graph;
    } else {
      GE_ASSERT_EQ(owner_graph.size(), 1U);
      *graph_has_inserted_to_set = *(owner_graph.begin());
      if (owner_graph.find(graph) == owner_graph.end()) {
        GELOGE(af::GRAPH_FAILED, "Node %s has diff owner graph %s with before nodes's graph %s", node->GetNamePtr(),
               graph->GetName().c_str(), (*graph_has_inserted_to_set)->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

graphStatus ExtractAndCheckInDataAnchorsByOrder(const std::vector<NodePtr> &nodes,
                                                std::vector<InDataAnchorPtr> &in_data_anchors,
                                                std::set<ComputeGraph *> &owner_graph) {
  in_data_anchors.clear();
  static ComputeGraph *graph_has_inserted_to_set = nullptr;
  for (const auto &node : nodes) {
    GE_ASSERT_NOTNULL(node);
    if (!CheckAndInsertGraph(node, owner_graph, &graph_has_inserted_to_set)) {
      return af::GRAPH_FAILED;
    }
    for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
      in_data_anchors.push_back(in_data_anchor);
    }
  }
  return af::GRAPH_SUCCESS;
}

graphStatus ExtractAndCheckOutDataAnchorsByOrder(const std::vector<NodePtr> &nodes,
                                                 std::vector<OutDataAnchorPtr> &out_data_anchors,
                                                 std::set<ComputeGraph *> &owner_graph) {
  out_data_anchors.clear();
  static ComputeGraph *graph_has_inserted_to_set = nullptr;
  for (const auto &node : nodes) {
    GE_ASSERT_NOTNULL(node);
    if (!CheckAndInsertGraph(node, owner_graph, &graph_has_inserted_to_set)) {
      return af::GRAPH_FAILED;
    }
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      out_data_anchors.push_back(out_data_anchor);
    }
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::InheritExecutionOrder(const std::vector<NodePtr> &new_nodes, const std::vector<NodePtr> &old_nodes,
                                  const ComputeGraphPtr &graph, bool need_convert_data_edges_2_ctrl_edges) {
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_TRUE(!old_nodes.empty());
  GE_ASSERT_TRUE(!new_nodes.empty());
  const auto &first_new_old = new_nodes.front();
  const auto &last_new_old = new_nodes.back();
  // 防止name重名
  const auto &noop_in = graph->AddNode(OpDescBuilder("noop_in_" + first_new_old->GetName(), NOOP).Build());
  const auto &noop_out = graph->AddNode(OpDescBuilder("noop_out_" + last_new_old->GetName(), NOOP).Build());
  // 注意old_nodes内部的控制关系不需要带到noop上
  NodeFilter node_filter = [&old_nodes](const Node &node) {
    return std::find(old_nodes.begin(), old_nodes.end(), node.shared_from_this()) == old_nodes.end();
  };
  for (const auto &old_node : old_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(CopyInCtrlEdges(old_node, noop_in, node_filter));
    GE_ASSERT_GRAPH_SUCCESS(CopyOutCtrlEdges(old_node, noop_out, node_filter));
    if (need_convert_data_edges_2_ctrl_edges) {
      GE_ASSERT_GRAPH_SUCCESS(ConvertInDataEdgesToInCtrlEdges(old_node, noop_in, node_filter));
      GE_ASSERT_GRAPH_SUCCESS(ConvertOutDataEdgesToOutCtrlEdges(old_node, noop_out, node_filter));
    }
  }
  if (noop_in->GetInControlNodesSize() > 0U) {
    for (const auto &new_node : new_nodes) {
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(noop_in->GetOutControlAnchor(), new_node->GetInControlAnchor()));
    }
  } else {
    GE_ASSERT_GRAPH_SUCCESS(RemoveJustNode(graph, noop_in));
  }
  if (noop_out->GetOutControlNodesSize() > 0U) {
    for (const auto &new_node : new_nodes) {
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(new_node->GetOutControlAnchor(), noop_out->GetInControlAnchor()));
    }
  } else {
    GE_ASSERT_GRAPH_SUCCESS(RemoveJustNode(graph, noop_out));
  }
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodesDataAnchors(const std::vector<NodePtr> &new_nodes, const std::vector<NodePtr> &old_nodes,
                                    const std::vector<int32_t> &inputs_map, const std::vector<int32_t> &outputs_map) {
  GE_ASSERT_GRAPH_SUCCESS(ReplaceNodesInDataAnchors(new_nodes, old_nodes, inputs_map));
  GE_ASSERT_GRAPH_SUCCESS(ReplaceNodesOutDataAnchors(new_nodes, old_nodes, outputs_map));
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodesInDataAnchors(const std::vector<NodePtr> &new_nodes, const std::vector<NodePtr> &old_nodes,
                                      const std::vector<int32_t> &inputs_map) {
  GE_ASSERT_TRUE(!old_nodes.empty());
  GE_ASSERT_TRUE(!new_nodes.empty());
  std::vector<InDataAnchorPtr> old_nodes_in_data_anchors;
  std::set<ComputeGraph *> owner_graph;
  GE_ASSERT_GRAPH_SUCCESS(ExtractAndCheckInDataAnchorsByOrder(old_nodes, old_nodes_in_data_anchors, owner_graph));
  std::vector<InDataAnchorPtr> new_nodes_in_data_anchors;
  GE_ASSERT_GRAPH_SUCCESS(ExtractAndCheckInDataAnchorsByOrder(new_nodes, new_nodes_in_data_anchors, owner_graph));
  GE_ASSERT_EQ(owner_graph.size(), 1U);
  return DoReplaceInDataAnchors(new_nodes_in_data_anchors, old_nodes_in_data_anchors, inputs_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodesOutDataAnchors(const std::vector<NodePtr> &new_nodes, const std::vector<NodePtr> &old_nodes,
                                       const std::vector<int32_t> &outputs_map) {
  GE_ASSERT_TRUE(!old_nodes.empty());
  GE_ASSERT_TRUE(!new_nodes.empty());
  std::vector<OutDataAnchorPtr> old_nodes_out_data_anchors;
  std::set<ComputeGraph *> owner_graph;
  GE_ASSERT_GRAPH_SUCCESS(ExtractAndCheckOutDataAnchorsByOrder(old_nodes, old_nodes_out_data_anchors, owner_graph));
  std::vector<OutDataAnchorPtr> new_nodes_out_data_anchors;
  GE_ASSERT_GRAPH_SUCCESS(ExtractAndCheckOutDataAnchorsByOrder(new_nodes, new_nodes_out_data_anchors, owner_graph));
  GE_ASSERT_EQ(owner_graph.size(), 1U);
  return ReplaceOutDataAnchors(new_nodes_out_data_anchors, old_nodes_out_data_anchors, outputs_map);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::CopyNodesInDataAnchors(const std::vector<NodePtr> &new_nodes, const std::vector<NodePtr> &old_nodes,
                                   const std::vector<int32_t> &inputs_map) {
  GE_ASSERT_TRUE(!old_nodes.empty());
  GE_ASSERT_TRUE(!new_nodes.empty());
  std::vector<InDataAnchorPtr> old_nodes_in_data_anchors;
  std::set<ComputeGraph *> owner_graph;
  GE_ASSERT_GRAPH_SUCCESS(ExtractAndCheckInDataAnchorsByOrder(old_nodes, old_nodes_in_data_anchors, owner_graph));
  std::vector<InDataAnchorPtr> new_nodes_in_data_anchors;
  GE_ASSERT_GRAPH_SUCCESS(ExtractAndCheckInDataAnchorsByOrder(new_nodes, new_nodes_in_data_anchors, owner_graph));
  GE_ASSERT_EQ(owner_graph.size(), 1U);
  return DoReplaceInDataAnchors(new_nodes_in_data_anchors, old_nodes_in_data_anchors, inputs_map, true);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyInCtrlEdges(const NodePtr &src_node,
                                                                                       const NodePtr &dst_node) {
  return CopyInCtrlEdges(src_node, dst_node, nullptr);
}

graphStatus AddCtrlEdges(const vector<std::shared_ptr<Node>> &src_nodes,
                         const vector<std::shared_ptr<Node>> &dst_nodes) {
  for (const auto &dst_node : dst_nodes) {
    GE_ASSERT_NOTNULL(dst_node);
    std::unordered_set<NodePtr> exist_in_ctrl_nodes_set;
    auto exist_in_ctrl_nodes = dst_node->GetInControlNodes();
    if (!exist_in_ctrl_nodes.empty()) {
      exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());
    }

    const auto dst_ctrl = dst_node->GetInControlAnchor();
    for (const auto &in_node : src_nodes) {
      GE_ASSERT_NOTNULL(in_node);
      if (exist_in_ctrl_nodes_set.count(in_node) > 0U) {
        continue;
      }
      const auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), dst_ctrl);
      if (ret != af::GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "Add ControlEdge from %s to %s failed", in_node->GetName().c_str(),
                             dst_node->GetName().c_str());
        GELOGE(af::GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed", in_node->GetName().c_str(),
               dst_node->GetName().c_str());
        return ret;
      }
    }
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyInCtrlEdges(const NodePtr &src_node,
                                                                                       const NodePtr &dst_node,
                                                                                       const NodeFilter &node_filter) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  const auto src_ctrl_in_nodes = NodeUtils::GetInControlNodes(*src_node, node_filter);
  if (src_ctrl_in_nodes.empty()) {
    return af::GRAPH_SUCCESS;
  }
  return AddCtrlEdges(src_ctrl_in_nodes, {dst_node});
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveInCtrlEdges(const NodePtr &src_node,
                                                                                       const NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return af::GRAPH_FAILED;
  }
  const auto ret = CopyInCtrlEdges(src_node, dst_node);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "[Copy][InCtrlEdges] failed, ret:%d, src_node:%s, dst_node:%s", ret,
           src_node->GetName().c_str(), dst_node->GetName().c_str());
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetInControlAnchor());
  src_node->GetInControlAnchor()->UnlinkAll();
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyOutCtrlEdges(const NodePtr &src_node,
                                                                                        const NodePtr &dst_node) {
  return CopyOutCtrlEdges(src_node, dst_node, nullptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyOutCtrlEdges(const NodePtr &src_node,
                                                                                        const NodePtr &dst_node,
                                                                                        const NodeFilter &node_filter) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node or dst_node is nullptr, check invalid");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return af::GRAPH_FAILED;
  }
  const auto &out_ctrl_nodes = NodeUtils::GetOutControlNodes(*src_node, node_filter);
  if (out_ctrl_nodes.empty()) {
    return af::GRAPH_SUCCESS;
  }
  return AddCtrlEdges({dst_node}, out_ctrl_nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveOutCtrlEdges(NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return af::GRAPH_FAILED;
  }
  const auto ret = CopyOutCtrlEdges(src_node, dst_node);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "[Copy][OutCtrlEdges] failed, ret:%d", ret);
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetOutControlAnchor());
  src_node->GetOutControlAnchor()->UnlinkAll();
  return af::GRAPH_SUCCESS;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ConvertInDataEdgesToInCtrlEdges(
    const NodePtr &src_node, const NodePtr &dst_node, const NodeFilter &node_filter) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  const auto src_in_data_nodes = NodeUtils::GetInDataNodes(*src_node, node_filter);
  if (src_in_data_nodes.empty()) {
    return af::GRAPH_SUCCESS;
  }
  return AddCtrlEdges(src_in_data_nodes, {dst_node});
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ConvertOutDataEdgesToOutCtrlEdges(
    const NodePtr &src_node, const NodePtr &dst_node, const NodeFilter &node_filter) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node or dst_node is nullptr, check invalid");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  const auto &src_out_data_nodes = NodeUtils::GetOutDataNodes(*src_node, node_filter);
  if (src_out_data_nodes.empty()) {
    return af::GRAPH_SUCCESS;
  }
  return AddCtrlEdges({dst_node}, src_out_data_nodes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveNodesToGraphAfterTargetNode(
    const ComputeGraphPtr &target_graph, const NodePtr &target_node, const ComputeGraphPtr &src_graph) {
  GE_ASSERT_NOTNULL(target_node);
  GE_ASSERT_NOTNULL(target_graph);
  GE_ASSERT_NOTNULL(src_graph);
  // 将expandGraph上的算子移动到targetGraph
  auto target_graph_impl = target_graph->impl_;
  GE_ASSERT_NOTNULL(target_graph_impl);
  auto target_iter = std::find(target_graph_impl->nodes_.begin(), target_graph_impl->nodes_.end(), target_node);
  GE_ASSERT_TRUE(target_iter != target_graph_impl->nodes_.end(), "Target node: %s should be in target graph: %s",
                 target_node->GetNamePtr(), target_graph->GetName().c_str());
  target_iter = next(target_iter);
  for (const auto &node : src_graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    // 输入输出算子不挪
    if ((node->GetType() == DATA) || (node->GetType() == NETOUTPUT)) {
      continue;
    }
    target_graph_impl->InsertToNodeList(target_iter, node);
    node->SetHostNode(target_graph_impl->is_valid_flag_);
    GE_ASSERT_SUCCESS(node->SetOwnerComputeGraph(target_graph), "SetOwnerComputeGraph:%s failed for node:%s",
                      target_graph->GetName().c_str(), node->GetNamePtr());
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const auto sub_graph_names = op_desc->GetSubgraphInstanceNames();
    for (size_t i = 0UL; i < sub_graph_names.size(); i++) {
      auto sub_graph = src_graph->GetSubgraph(sub_graph_names[i]);
      GE_ASSERT_NOTNULL(sub_graph);
      sub_graph->SetParentGraph(target_graph);
    }
  }
  // 将expandGraph中的剩余子图继承到根图
  const auto root_graph = GraphUtils::FindRootGraph(target_graph);
  for (const auto &subgraph : src_graph->GetAllSubgraphs()) {
    root_graph->AddSubGraph(subgraph);
  }
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ExpandNodeWithGraph(const NodePtr &target_node, const ComputeGraphPtr &expand_graph) {
  GE_ASSERT_NOTNULL(target_node);
  auto target_graph = target_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(target_graph);
  // 搬移节点
  GE_ASSERT_SUCCESS(GraphUtils::MoveNodesToGraphAfterTargetNode(target_graph, target_node, expand_graph));
  // 处理输入连边关系
  for (const auto &input_node : expand_graph->GetInputNodes()) {
    GE_ASSERT_SUCCESS(ReLinkInputDataEdge(input_node, target_node));
    NodeUtils::UnlinkAll(*input_node);
  }
  // 处理输出连边关系
  // 如果expand图存在netoutput，则先断开netoutput
  const auto net_output_node = expand_graph->FindFirstNodeMatchType(NETOUTPUT);
  const auto sub_graph_out_node_info = expand_graph->GetGraphOutNodesInfo();
  if (net_output_node != nullptr) {
    NodeUtils::UnlinkAll(*net_output_node);
  }
  const auto out_nodes_info = expand_graph->GetGraphOutNodesInfo();
  for (size_t index = 0UL; index < out_nodes_info.size(); index++) {
    GE_ASSERT_SUCCESS(
        RelinkOutputNodeEdge(out_nodes_info[index].first, out_nodes_info[index].second, target_node, index));
  }
  // 处理输出信息的映射关系
  const auto target_graph_out_node_info = target_graph->GetGraphOutNodesInfo();

  std::vector<std::pair<NodePtr, int32_t>> new_output_info;
  for (const auto &out_node_info : target_graph_out_node_info) {
    if (out_node_info.first == target_node) {
      GE_ASSERT_TRUE(static_cast<size_t>(out_node_info.second) < sub_graph_out_node_info.size(),
                     "out node index:%zu out of range, sub_graph_out_node_info size:%zu",
                     static_cast<size_t>(out_node_info.second), sub_graph_out_node_info.size());
      (void)new_output_info.emplace_back(sub_graph_out_node_info[out_node_info.second]);
    } else {
      (void)new_output_info.emplace_back(out_node_info);
    }
  }
  // 删除原算子
  NodeUtils::UnlinkAll(*target_node);
  GE_ASSERT_SUCCESS(RemoveNodeWithoutRelink(target_graph, target_node));
  GE_ASSERT_SUCCESS(target_graph->SetGraphOutNodesInfo(new_output_info));
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AppendInputNode(const ComputeGraphPtr &graph,
                                                                                       const NodePtr &node) {
  if (graph->AddInputNode(node) == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "AddInputNode %s(%s) failed, graph:%s", node->GetName().c_str(),
                         node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(af::GRAPH_FAILED, "[Add][InputNode] %s(%s) failed, graph:%s", node->GetName().c_str(),
           node->GetType().c_str(), graph->GetName().c_str());
    return af::GRAPH_FAILED;
  }
  graph->SetInputSize(graph->GetInputSize() + 1U);
  if (graph->impl_ == nullptr) {
    GELOGE(af::GRAPH_FAILED, "Graph impl is nullptr.");
    return af::GRAPH_FAILED;
  }
  graph->impl_->inputs_order_.emplace_back(node->GetName());
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr GraphUtils::FindRootGraph(ComputeGraphPtr graph) {
  ComputeGraphPtr result = nullptr;
  while (graph != nullptr) {
    result = std::move(graph);
    graph = result->GetParentGraph();
  }
  return result;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyComputeGraph(
    const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter, const GraphFilter &graph_filter,
    const AttrFilter &attr_filter, ComputeGraphPtr &dst_compute_graph) {
  GE_CHECK_NOTNULL(src_compute_graph);
  if (src_compute_graph->GetParentGraph() != nullptr) {
    GELOGE(af::GRAPH_FAILED,
           "[Check][RootGraph] Only support copy root graph, current graph name:%s, "
           "parent graph name:%s.",
           src_compute_graph->GetName().c_str(), src_compute_graph->GetParentGraph()->GetName().c_str());
    return af::GRAPH_FAILED;
  }

  const int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> old_2_new_node;
  std::map<ConstOpDescPtr, OpDescPtr> old_2_new_op_desc;
  const graphStatus ret = CopyComputeGraph(src_compute_graph, node_filter, graph_filter, attr_filter, dst_compute_graph,
                                           old_2_new_node, old_2_new_op_desc, depth);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "[Copy][ComputeGraphPtr] failed, ret:%d.", ret);
    return af::GRAPH_FAILED;
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr GraphUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc) {
  GE_CHECK_NOTNULL_EXEC(org_op_desc, return nullptr);
  const auto op_def = ComGraphMakeShared<proto::OpDef>();
  GE_CHECK_NOTNULL_EXEC(op_def, return nullptr);

  ModelSerializeImp imp;
  (void)imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  GE_CHK_BOOL_EXEC(imp.UnserializeOpDesc(op_desc, *op_def), REPORT_INNER_ERR_MSG("E18888", "UnserializeOpDesc failed");
                   return op_desc, "[Call][UnserializeOpDesc] op_desc unserialize failed");

  GE_CHECK_NOTNULL_EXEC(op_desc->impl_, return nullptr);
  op_desc->ext_attrs_ = org_op_desc->ext_attrs_;

  // This function may be called by some passes of fusion engine, in this condition, do not need these attribute
  if (!op_desc->impl_->input_name_idx_.empty()) {
    op_desc->impl_->input_name_idx_.clear();
  }
  if (!op_desc->impl_->output_name_idx_.empty()) {
    op_desc->impl_->output_name_idx_.clear();
  }
  op_desc->impl_->MutableIRMeta() = IRMetaData(op_desc->GetName());
  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr GraphUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc,
                                                                                const AttrFilter &attr_filter) {
  GE_ASSERT_NOTNULL(org_op_desc);
  GE_ASSERT_NOTNULL(org_op_desc->impl_);
  const auto op_def = ComGraphMakeShared<proto::OpDef>();
  GE_ASSERT_NOTNULL(op_def);

  ModelSerializeImp imp;
  (void)imp.SerializeOpDesc(org_op_desc, op_def.get());
  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  GE_ASSERT_TRUE(imp.UnserializeOpDesc(op_desc, *op_def));
  // weight's data call `Clone` for deep copy if needed
  if (ConstantUtils::IsConstant(op_desc) && ((attr_filter == nullptr) || attr_filter(*op_desc, ATTR_NAME_WEIGHTS))) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(org_op_desc, ATTR_NAME_WEIGHTS, weight)) {
      const GeTensor copy_weight = weight->Clone();
      GE_ASSERT_TRUE(AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, copy_weight));
      GELOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", op_desc->GetName().c_str());
    }
  }
  // remove attr by attr_filter
  for (const auto &attr_name : op_desc->GetAllAttrNames()) {
    if ((attr_filter != nullptr) && (!attr_filter(*op_desc, attr_name))) {
      GE_ASSERT_GRAPH_SUCCESS(op_desc->DelAttr(attr_name));
    }
  }
  GE_ASSERT_NOTNULL(op_desc->impl_);
  op_desc->ext_attrs_ = org_op_desc->ext_attrs_;
  op_desc->impl_->input_name_idx_.insert(org_op_desc->impl_->input_name_idx_.cbegin(),
                                         org_op_desc->impl_->input_name_idx_.cend());
  op_desc->impl_->MutableIRMeta() = org_op_desc->impl_->GetIRMeta();
  op_desc->impl_->output_name_idx_.insert(org_op_desc->impl_->output_name_idx_.cbegin(),
                                          org_op_desc->impl_->output_name_idx_.cend());

  op_desc->impl_->infer_func_ = org_op_desc->impl_->infer_func_;
  op_desc->impl_->infer_format_func_ = org_op_desc->impl_->infer_format_func_;
  op_desc->impl_->verifier_func_ = org_op_desc->impl_->verifier_func_;

  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr GraphUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  return CopyOpDesc(org_op_desc, nullptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph) {
  return CopyComputeGraph(src_compute_graph, nullptr, nullptr, nullptr, dst_compute_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph,
                             std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                             std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, const int32_t depth) {
  return CopyComputeGraph(src_compute_graph, nullptr, nullptr, nullptr, dst_compute_graph, node_old_2_new,
                          op_desc_old_2_new, depth);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyOpAndSubgraph(
    const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter, const GraphFilter &graph_filter,
    const AttrFilter &attr_filter, ComputeGraphPtr &dst_compute_graph, std::map<ConstNodePtr, NodePtr> &node_old_2_new,
    std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, std::unordered_map<std::string, NodePtr> &all_new_nodes,
    const int32_t depth) {
  GE_CHECK_NOTNULL(src_compute_graph);
  GE_CHECK_NOTNULL(dst_compute_graph);
  const auto dst_root_compute_graph = FindRootGraph(dst_compute_graph);
  GE_CHECK_NOTNULL(dst_root_compute_graph);
  const auto src_root_compute_graph = FindRootGraph(src_compute_graph);
  GE_CHECK_NOTNULL(src_root_compute_graph);
  for (const auto &n : src_compute_graph->GetDirectNode()) {
    if ((node_filter != nullptr) && (!node_filter(*n))) {
      continue;
    }
    const auto &op_desc = GraphUtils::CopyOpDesc(n->GetOpDesc(), attr_filter);
    GE_CHECK_NOTNULL(op_desc);
    GE_CHECK_NOTNULL(op_desc->impl_);
    op_desc->SetName(n->GetName());
    op_desc->impl_->MutableIRMeta() = n->GetOpDesc()->impl_->GetIRMeta();
    op_desc->impl_->subgraph_names_to_index_ = n->GetOpDesc()->impl_->subgraph_names_to_index_;
    op_desc->impl_->subgraph_instance_names_ = n->GetOpDesc()->impl_->subgraph_instance_names_;

    const NodePtr node = dst_compute_graph->AddNode(op_desc, n->GetOpDesc()->GetId());
    if (node == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "AddNode %s to graph:%s failed", op_desc->GetName().c_str(),
                           dst_compute_graph->GetName().c_str());
      GELOGE(af::GRAPH_FAILED, "[Add][Node][%s] to graph:%s failed", op_desc->GetName().c_str(),
             dst_compute_graph->GetName().c_str());
      return af::GRAPH_FAILED;
    }
    all_new_nodes[node->GetName()] = node;
    node_old_2_new[n] = node;
    op_desc_old_2_new[n->GetOpDesc()] = op_desc;

    // copy subgraph from old graph to new graph
    const auto &subgraph_names = n->GetOpDesc()->GetSubgraphInstanceNames();
    const size_t subgraph_num = subgraph_names.size();
    for (size_t subgraph_idx = 0U; subgraph_idx < subgraph_num; ++subgraph_idx) {
      const auto &subgraph_name = subgraph_names[subgraph_num - 1U - subgraph_idx];
      const auto src_subgraph = src_root_compute_graph->GetSubgraph(subgraph_name);
      if ((src_subgraph == nullptr) && subgraph_name.empty()) {
        GELOGD("node=%s subgraph is empty, subgraph_idx=%zu, subgraph_num=%zu.", n->GetName().c_str(), subgraph_idx,
               subgraph_num);
        continue;
      }
      GE_CHECK_NOTNULL(src_subgraph, ", get subgraph[%s] failed, node=%s.", subgraph_name.c_str(),
                       n->GetName().c_str());
      if ((graph_filter != nullptr) &&
          (!graph_filter(*src_subgraph->GetParentNode(), src_subgraph->GetName().c_str(), src_subgraph))) {
        op_desc->RemoveSubgraphInstanceName(subgraph_name);
        continue;
      }
      ComputeGraphPtr dst_subgraph = ComGraphMakeShared<ComputeGraph>(src_subgraph->GetName());
      GE_CHECK_NOTNULL(dst_subgraph);
      dst_subgraph->SetParentGraph(dst_compute_graph);
      std::map<ConstNodePtr, NodePtr> sub_node_old_2_new;
      std::map<ConstOpDescPtr, OpDescPtr> sub_op_desc_old_2_new;
      const graphStatus ret = CopyComputeGraph(src_subgraph, node_filter, graph_filter, attr_filter, dst_subgraph,
                                               sub_node_old_2_new, sub_op_desc_old_2_new, depth + 1);
      if (ret != af::GRAPH_SUCCESS) {
        GELOGE(af::GRAPH_FAILED, "[Copy][SubGraph] %s of parent node:%s failed.", src_subgraph->GetName().c_str(),
               node->GetName().c_str());
        return af::GRAPH_FAILED;
      }
      (void)dst_root_compute_graph->AddSubGraph(dst_subgraph);
      dst_subgraph->SetParentNode(node);
    }
  }
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyComputeGraph(
    const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter, const GraphFilter &graph_filter,
    const AttrFilter &attr_filter, ComputeGraphPtr &dst_compute_graph, std::map<ConstNodePtr, NodePtr> &node_old_2_new,
    std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, const int32_t depth) {
  GE_CHECK_NOTNULL(dst_compute_graph);
  GE_CHECK_NOTNULL(src_compute_graph);

  if (depth >= kCopyGraphMaxRecursionDepth) {
    REPORT_INNER_ERR_MSG("E18888", "param depth:%d >= %d(allow max subgraphs)", depth, kCopyGraphMaxRecursionDepth);
    GELOGE(af::GRAPH_FAILED, "[Check][Param] too many subgraphs exist: %d > %d (allow max subgraphs)", depth,
           kCopyGraphMaxRecursionDepth);
    return af::GRAPH_FAILED;
  }
  // copy op and subgraph from old graph to new graph
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  graphStatus ret = CopyOpAndSubgraph(src_compute_graph, node_filter, graph_filter, attr_filter, dst_compute_graph,
                                      node_old_2_new, op_desc_old_2_new, all_new_nodes, depth);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "[Copy][OpAndSubGraph] failed.");
    return af::GRAPH_FAILED;
  }

  for (const auto &n : src_compute_graph->GetDirectNode()) {
    if ((node_filter != nullptr) && (!node_filter(*n))) {
      continue;
    }
    if (RelinkGraphEdges(n, "", all_new_nodes) != af::GRAPH_SUCCESS) {
      GELOGE(af::GRAPH_FAILED, "[Relink][Edges] failed.");
      return af::GRAPH_FAILED;
    }
  }
  // To keep subgraph consistent with the source graph
  std::vector<ComputeGraphPtr> new_subgraphs;
  const auto old_subgraphs = src_compute_graph->GetAllSubgraphs();
  for (const auto &sub_graph : old_subgraphs) {
    if ((graph_filter != nullptr) &&
        (!graph_filter(*sub_graph->GetParentNode(), sub_graph->GetName().c_str(), sub_graph))) {
      continue;
    }
    const auto new_subgraph = dst_compute_graph->GetSubgraph(sub_graph->GetName());
    GE_CHECK_NOTNULL(new_subgraph);
    GELOGD("Copy new subgraph:%s.", sub_graph->GetName().c_str());
    new_subgraphs.push_back(new_subgraph);
  }
  dst_compute_graph->SetAllSubgraphs(new_subgraphs);

  // copy members from old graph to new graph
  ret = CopyMembers(src_compute_graph, dst_compute_graph, all_new_nodes);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGE(af::GRAPH_FAILED, "[Copy][Members] failed, ret:%d.", ret);
    return af::GRAPH_FAILED;
  }

  // inherit all attr from old graph to new graph
  InheritOriginalAttr(src_compute_graph, dst_compute_graph);

  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::CopyMembers(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph,
                        const std::unordered_map<std::string, NodePtr> &all_new_nodes) {
  if ((src_compute_graph == nullptr) || (src_compute_graph->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "Param src_compute_graph is nullptr, check invalid");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Src compute graph is nullptr.");
    return af::GRAPH_FAILED;
  }
  if ((dst_compute_graph == nullptr) || (dst_compute_graph->impl_ == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "Param dst_compute_graph is nullptr, check invalid");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Dst compute graph is nullptr.");
    return af::GRAPH_FAILED;
  }
  // copy info of output nodes from old graph to new graph.
  const std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info = src_compute_graph->GetGraphOutNodesInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  for (const auto &info : out_nodes_info) {
    const auto it = all_new_nodes.find(info.first->GetName());
    if (it == all_new_nodes.end()) {
      GELOGW("[Check][Param] Find output node:%s failed.", info.first->GetName().c_str());
      continue;
    }
    new_out_nodes_info.emplace_back(it->second, info.second);
  }
  GE_ASSERT_SUCCESS(dst_compute_graph->SetGraphOutNodesInfo(new_out_nodes_info, false));

  // copy info of input nodes from old graph to new graph.
  const ComputeGraph::Vistor<NodePtr> &input_nodes = src_compute_graph->GetInputNodes();
  for (const auto &node : input_nodes) {
    const auto it = all_new_nodes.find(node->GetName());
    if (it == all_new_nodes.end()) {
      GELOGW("[Check][Param] Find input node:%s failed.", node->GetName().c_str());
      continue;
    }
    (void)dst_compute_graph->AddInputNode(it->second);
  }

  // copy target info nodes from old graph to new graph.
  const std::vector<NodePtr> &src_traget_nodes_info = src_compute_graph->GetGraphTargetNodesInfo();
  std::vector<NodePtr> dst_traget_nodes_info;
  for (const auto &node : src_traget_nodes_info) {
    const auto it = all_new_nodes.find(node->GetName());
    if (it == all_new_nodes.end()) {
      GELOGW("[Check][Param] Find target info node:%s failed.", node->GetName().c_str());
      continue;
    }
    dst_traget_nodes_info.emplace_back(it->second);
  }
  dst_compute_graph->SetGraphTargetNodesInfo(dst_traget_nodes_info);

  // graph属性序列化
  dst_compute_graph->impl_->attrs_ = src_compute_graph->impl_->attrs_;

  // copy other members from old graph to new graph.
  dst_compute_graph->impl_->data_format_ = src_compute_graph->impl_->data_format_;
  dst_compute_graph->impl_->need_iteration_ = src_compute_graph->impl_->need_iteration_;
  dst_compute_graph->impl_->is_summary_graph_ = src_compute_graph->impl_->is_summary_graph_;
  dst_compute_graph->impl_->is_valid_flag_ = src_compute_graph->impl_->is_valid_flag_;
  dst_compute_graph->impl_->input_size_ = src_compute_graph->impl_->input_size_;
  dst_compute_graph->impl_->output_size_ = src_compute_graph->impl_->output_size_;
  dst_compute_graph->impl_->inputs_order_ = src_compute_graph->impl_->inputs_order_;
  dst_compute_graph->impl_->op_name_map_ = src_compute_graph->impl_->op_name_map_;
  dst_compute_graph->impl_->out_nodes_map_ = src_compute_graph->impl_->out_nodes_map_;
  dst_compute_graph->impl_->params_share_map_ = src_compute_graph->impl_->params_share_map_;
  dst_compute_graph->impl_->graph_id_ = src_compute_graph->impl_->graph_id_;
  return af::GRAPH_SUCCESS;
}

/// Make a copy of ComputeGraph.
/// @param graph: original graph.
/// @param suffix: node name suffix of new graph.
/// @param output_nodes: output nodes of new graph.
/// @return ComputeGraphPtr
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr
GraphUtils::CloneGraph(const ComputeGraphPtr &graph, const std::string &suffix, std::vector<NodePtr> &input_nodes,
                       std::vector<NodePtr> &output_nodes) {
  GE_CHK_BOOL_EXEC(graph != nullptr, REPORT_INNER_ERR_MSG("E18888", "param graph is nullptr, check invalid.");
                   return nullptr, "[Check][Param] Original graph is null");
  ComputeGraphPtr new_graph = ComGraphMakeShared<ComputeGraph>(graph->GetName());
  GE_CHK_BOOL_EXEC(new_graph != nullptr,
                   REPORT_INNER_ERR_MSG("E18888", "create computegraph %s failed.", graph->GetName().c_str());
                   return nullptr, "[Create][ComputeGraph] %s failed", graph->GetName().c_str());

  std::unordered_map<std::string, NodePtr> all_new_nodes;
  for (const auto &n : graph->GetDirectNode()) {
    const OpDescPtr op_desc = GraphUtils::CopyOpDesc(n->GetOpDesc());
    GE_CHK_BOOL_EXEC(op_desc != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "Create node:%s failed.", n->GetOpDesc()->GetName().c_str());
                     return nullptr, "[Create][Node] %s failed", n->GetOpDesc()->GetName().c_str());

    if (CopyTensorAttrs(op_desc, n) != af::GRAPH_SUCCESS) {
      return nullptr;
    }

    const bool is_const_op = (n->GetType() == CONSTANT) || (n->GetType() == CONSTANTOP);
    if (is_const_op) {
      GeTensorPtr weight = nullptr;
      if (!AttrUtils::MutableTensor(n->GetOpDesc(), ATTR_NAME_WEIGHTS, weight)) {
        GELOGI("Can not find attr ATTR_NAME_WEIGHTS for node:%s.", n->GetName().c_str());
        continue;
      }
      const GeTensor copy_weight = weight->Clone();
      if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, copy_weight)) {
        REPORT_INNER_ERR_MSG("E18888", "Clone ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Tensor] Clone ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
        return nullptr;
      }
      GELOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", op_desc->GetName().c_str());
    }

    op_desc->SetName(n->GetName() + suffix);
    NodePtr node = new_graph->AddNode(op_desc);
    GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERR_MSG("E18888", "add node %s to graph:%s failed",
                                                           op_desc->GetName().c_str(), new_graph->GetName().c_str());
                     return nullptr, "[Add][Node] [%s] to graph:%s failed", op_desc->GetName().c_str(),
                            new_graph->GetName().c_str());
    all_new_nodes[node->GetName()] = node;

    if (OpTypeUtils::IsDataNode(node->GetType())) {
      input_nodes.emplace_back(node);
    } else if (node->GetType() == NETOUTPUT) {
      output_nodes.emplace_back(node);
    } else {
      // do nothing
    }
  }

  for (const auto &n : graph->GetDirectNode()) {
    if (RelinkGraphEdges(n, suffix, all_new_nodes) != af::GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  // inherit all attr from old graph to new graph
  InheritOriginalAttr(graph, new_graph);

  // copy info of output nodes from old graph to new graph.
  const std::vector<std::pair<NodePtr, int32_t>> out_nodes_info = graph->GetGraphOutNodesInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  for (const auto &info : out_nodes_info) {
    const auto it = all_new_nodes.find(info.first->GetName());
    if (it != all_new_nodes.end()) {
      new_out_nodes_info.emplace_back(it->second, info.second);
    }
  }
  GE_ASSERT_SUCCESS(new_graph->SetGraphOutNodesInfo(new_out_nodes_info, false));
  return new_graph;
}

/// Copy tensor attribute to new node.
/// @param [in] dst_node: cloned node.
/// @param [in] src_node: original node.
/// @return success: GRAPH_SUCCESS
graphStatus GraphUtils::CopyTensorAttrs(const OpDescPtr &dst_desc, const NodePtr &src_node) {
  if (dst_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param dst_desc is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Input param dst node not valid");
    return af::GRAPH_FAILED;
  }
  if ((src_node == nullptr) || (src_node->GetOpDesc() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param src_node is nullptr or its opdesc is nullptr, check invalid.");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Input param src node not valid");
    return af::GRAPH_FAILED;
  }

  const auto &src_desc = src_node->GetOpDesc();
  dst_desc->CopyAttrsFrom(*src_desc);

  for (uint32_t i = 0U; i < src_node->GetAllInDataAnchorsSize(); ++i) {
    const auto input_desc = dst_desc->MutableInputDesc(i);
    if (input_desc == nullptr) {
      continue;
    }
    input_desc->CopyAttrsFrom(src_desc->GetInputDesc(i));
  }

  for (uint32_t i = 0U; i < src_node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_desc = dst_desc->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Param dst node:%s not valid, output_desc[%u] is nullptr",
                           dst_desc->GetName().c_str(), i);
      GELOGE(af::GRAPH_FAILED, "[Check][Param] Param dst node:%s not valid", dst_desc->GetName().c_str());
      return af::GRAPH_FAILED;
    }
    output_desc->CopyAttrsFrom(src_desc->GetOutputDesc(i));
  }

  return af::GRAPH_SUCCESS;
}

graphStatus RelinkClonedDataEdges(const NodePtr &node, const NodePtr &new_node, const NodeCloneMap &node_clone_map) {
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", node->GetName().c_str());
                     return af::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s",
                            node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      const auto *peer_node = peer_in_anchor->GetOwnerNodeBarePtr();
      GE_CHK_BOOL_EXEC(peer_node != nullptr,
                       REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", node->GetName().c_str());
                       return af::GRAPH_FAILED, "Peer in node:%s is null", node->GetName().c_str());
      const auto peer_it = node_clone_map.find(peer_node);
      if (peer_it == node_clone_map.end()) {
        REPORT_INNER_ERR_MSG("E18888", "node_clone_map does not contain peer node:%s.",
                             peer_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(af::GRAPH_FAILED, "[Check][Param] peer node[%s] not found",
               peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return af::GRAPH_FAILED;
      }
      const auto ret = GraphUtils::AddEdge(new_node->GetOutAnchor(out_anchor->GetIdx()),
                                           peer_it->second->GetInAnchor(peer_in_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == af::GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "add data edge from %s to %s failed", new_node->GetName().c_str(),
                                            peer_it->second->GetName().c_str());
                       return af::GRAPH_FAILED, "[Invoke][AddEdge] link data edge failed[%s to %s]",
                              new_node->GetName().c_str(), peer_it->second->GetName().c_str());
    }
  }
  return af::GRAPH_SUCCESS;
}

graphStatus RelinkClonedControlEdges(const NodePtr &node, const NodePtr &new_node, const NodeCloneMap &node_clone_map) {
  if (node->GetOutControlAnchor() != nullptr) {
    for (const auto peer_in_control_anchor : node->GetOutControlAnchor()->GetPeerAnchorsPtr()) {
      GE_CHECK_NOTNULL(peer_in_control_anchor);
      const auto *peer_node = peer_in_control_anchor->GetOwnerNodeBarePtr();
      GE_CHK_BOOL_EXEC(peer_node != nullptr, REPORT_INNER_ERR_MSG("E18888", "Peer out node is null");
                       return af::GRAPH_FAILED, "[Invoke][GetOwnerNode] Peer out node is null");
      const auto peer_it = node_clone_map.find(peer_node);
      if (peer_it == node_clone_map.end()) {
        REPORT_INNER_ERR_MSG("E18888", "node_clone_map does not contain peer node:%s.",
                             peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(af::GRAPH_FAILED, "[Check][Param] peer node[%s] not found",
               peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        return af::GRAPH_FAILED;
      }
      const auto ret = GraphUtils::AddEdge(new_node->GetOutControlAnchor(),
                                           peer_it->second->GetInAnchor(peer_in_control_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == af::GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "add control edge from %s to %s failed.",
                                            new_node->GetName().c_str(), peer_it->second->GetName().c_str());
                       return af::GRAPH_FAILED, "[Invoke][AddEdge] link control edge failed[%s to %s]",
                              new_node->GetName().c_str(), peer_it->second->GetName().c_str());
    }
  }
  return af::GRAPH_SUCCESS;
}

/// Relink all edges for cloned ComputeGraph.
/// @param [in] node: original node.
/// @param [in] node_clone_map: source-node to cloned-node mapping.
/// @return success: GRAPH_SUCCESS
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RelinkGraphEdges(const NodePtr &node, const NodeCloneMap &node_clone_map) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr or its opdesc is nullptr. check invalid");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Input node not valid");
    return af::GRAPH_FAILED;
  }
  const auto node_it = node_clone_map.find(node.get());
  if (node_it == node_clone_map.end()) {
    REPORT_INNER_ERR_MSG("E18888", "node_clone_map does not contain node:%s.", node->GetName().c_str());
    GELOGE(af::GRAPH_FAILED, "[Check][Param] node[%s] not found", node->GetName().c_str());
    return af::GRAPH_FAILED;
  }
  GE_CHK_STATUS_RET(RelinkClonedDataEdges(node, node_it->second, node_clone_map), "Failed to relink data edges");
  return RelinkClonedControlEdges(node, node_it->second, node_clone_map);
}

/// Relink all edges for cloned ComputeGraph.
/// @param [in] node: original node.
/// @param [in] suffix: node name suffix of new node.
/// @param [in] all_nodes: all nodes in new graph.
/// @return success: GRAPH_SUCCESS
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RelinkGraphEdges(
    const NodePtr &node, const std::string &suffix, const std::unordered_map<std::string, NodePtr> &all_nodes) {
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    REPORT_INNER_ERR_MSG("E18888", "param node is nullptr or its opdesc is nullptr. check invalid");
    GELOGE(af::GRAPH_FAILED, "[Check][Param] Input node not valid");
    return af::GRAPH_FAILED;
  }

  auto it = all_nodes.find(node->GetName() + suffix);
  if (it == all_nodes.end()) {
    REPORT_INNER_ERR_MSG("E18888", "all_nodes does not contain node:%s.", node->GetName().c_str());
    GELOGE(af::GRAPH_FAILED, "[Check][Param] node[%s] not found", node->GetName().c_str());
    return af::GRAPH_FAILED;
  }
  const auto &new_node = it->second;

  // traversing from the parent node can be completely restored in the original one-to-many order.
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERR_MSG("E18888", "out data anchor is null, node:%s.", node->GetName().c_str());
                     return af::GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s",
                            node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNodeBarePtr() != nullptr,
                       REPORT_INNER_ERR_MSG("E18888", "Peer in node:%s is null", node->GetName().c_str());
                       return af::GRAPH_FAILED, "Peer in node:%s is null", node->GetName().c_str());
      it = all_nodes.find(peer_in_anchor->GetOwnerNodeBarePtr()->GetName() + suffix);
      if (it == all_nodes.end()) {
        GELOGW("[Check][Param] node[%s] not found", peer_in_anchor->GetOwnerNode()->GetName().c_str());
        continue;
      }
      const auto &new_peer_in_node = it->second;
      const auto ret = GraphUtils::AddEdge(new_node->GetOutAnchor(out_anchor->GetIdx()),
                                           new_peer_in_node->GetInAnchor(peer_in_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == af::GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "add data edge from %s to %s failed", new_node->GetName().c_str(),
                                            new_peer_in_node->GetName().c_str());
                       return af::GRAPH_FAILED, "[Invoke][AddEdge] link data edge failed[%s to %s]",
                              new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
    }
  }

  if (node->GetOutControlAnchor() != nullptr) {
    for (const auto peer_in_control_anchor : node->GetOutControlAnchor()->GetPeerAnchorsPtr()) {
      GE_CHECK_NOTNULL(peer_in_control_anchor);
      GE_CHK_BOOL_EXEC(peer_in_control_anchor->GetOwnerNodeBarePtr() != nullptr,
                       REPORT_INNER_ERR_MSG("E18888", "Peer out node is null");
                       return af::GRAPH_FAILED, "[Invoke][GetOwnerNode] Peer out node is null");
      it = all_nodes.find(peer_in_control_anchor->GetOwnerNodeBarePtr()->GetName() + suffix);
      if (it == all_nodes.end()) {
        GELOGW("[Check][Param] node[%s] not found", peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        continue;
      }
      const auto &new_peer_in_node = it->second;
      const auto ret = GraphUtils::AddEdge(new_node->GetOutControlAnchor(),
                                           new_peer_in_node->GetInAnchor(peer_in_control_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == af::GRAPH_SUCCESS,
                       REPORT_INNER_ERR_MSG("E18888", "add control edge from %s to %s failed.",
                                            new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
                       return af::GRAPH_FAILED, "[Invoke][AddEdge] link control edge failed[%s to %s]",
                              new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
    }
  }
  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::GetRefMapping(const ComputeGraphPtr &graph, SymbolToAnchors &symbol_to_anchors,
                                      AnchorToSymbol &anchor_to_symbol) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    // in_data_anchor
    GE_ASSERT_GRAPH_SUCCESS(HandleInAnchorMapping(graph, node, symbol_to_anchors, anchor_to_symbol),
                            "Find ref_mapping for in_data_anchors of node %s failed.", node->GetName().c_str());
    // out_data_anchor
    GE_ASSERT_GRAPH_SUCCESS(HandleOutAnchorMapping(node, symbol_to_anchors, anchor_to_symbol),
                            "Find ref_mapping for out_data_anchors of node %s failed.", node->GetName().c_str());
  }
  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::HandleInAnchorMapping(const ComputeGraphPtr &graph, const NodePtr &node,
                                              SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  if (node->GetOwnerComputeGraph() != graph) {
    // when curr graph is subgraph , to handle subgraph input/output ref mapping
    if (NodeUtils::IsSubgraphOutput(node)) {
      return HandleSubgraphOutput(node, symbol_to_anchors, anchor_to_symbol);
    }

    if (NodeUtils::IsSubgraphInput(node)) {
      return HandleSubgraphInput(node, symbol_to_anchors, anchor_to_symbol);
    }
  }

  const std::string &type = node->GetType();
  if ((type == MERGE) || (type == STREAMMERGE)) {
    return HandleMergeInput(node, symbol_to_anchors, anchor_to_symbol);
  }

  for (const auto in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    const NodeIndexIO cur_node_info(node, in_data_anchor->GetIdx(), kIn);
    const OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      const std::string &symbol = cur_node_info.ToString();
      GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
      symbol_to_anchors[symbol] = {cur_node_info};
      anchor_to_symbol[symbol] = symbol;
    } else {
      const NodeIndexIO exist_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
      GE_ASSERT_GRAPH_SUCCESS(UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol));
    }
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::HandleOutAnchorMapping(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                               AnchorToSymbol &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    const NodeIndexIO cur_node_info(node, out_data_anchor->GetIdx(), kOut);
    if (anchor_to_symbol.find(cur_node_info.ToString()) != anchor_to_symbol.end()) {
      continue;
    }

    NodePtr refed_node = nullptr;
    bool is_ref_from_other = false;
    GE_ASSERT_GRAPH_SUCCESS(CheckIsRefFromOther(out_data_anchor, refed_node, is_ref_from_other));
    NodeIndexIO exist_ref_data_info(refed_node, 0U, kOut);
    if (is_ref_from_other && (anchor_to_symbol.find(exist_ref_data_info.ToString()) != anchor_to_symbol.end())) {
      GELOGD("Node %s output:%d is ref from node: %s.", node->GetName().c_str(), out_data_anchor->GetIdx(),
             exist_ref_data_info.ToString().c_str());
      GE_ASSERT_GRAPH_SUCCESS(
          UpdateRefMapping(cur_node_info, exist_ref_data_info, symbol_to_anchors, anchor_to_symbol));
    }

    // 这里ref from input和ref from refdata不冲突
    int32_t reuse_in_index = -1;
    const bool reuse_input_flag = IsRefFromInput(out_data_anchor, reuse_in_index);
    if (reuse_input_flag && (node->GetInDataAnchor(reuse_in_index) != nullptr)) {
      const NodeIndexIO exist_node_info(node, reuse_in_index, kIn);
      if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != af::GRAPH_SUCCESS) {
        GE_LOGE("[Update][SymbolMapping] failed.");
        return af::GRAPH_FAILED;
      }
    } else {
      if (reuse_input_flag) {
        GELOGW(
            "[GetRefMapping][Check] Invalid reuse_input attr on output %d of node %s, please check attr reuse_input "
            "and reuse_input_index",
            out_data_anchor->GetIdx(), node->GetName().c_str());
      }
      const std::string &symbol = cur_node_info.ToString();
      GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
      (void)symbol_to_anchors.emplace(std::make_pair(symbol, std::list<NodeIndexIO>{cur_node_info}));
      (void)anchor_to_symbol.emplace(std::make_pair(symbol, symbol));
    }
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::HandleSubgraphInput(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                            AnchorToSymbol &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  // Data in subgraph
  uint32_t index = 0U;
  if (!af::AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index)) {
    REPORT_INNER_ERR_MSG("E18888", "Get Attr ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    GE_LOGE("[Get][Attr] ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    return af::GRAPH_FAILED;
  }
  const NodePtr parent_node = node->GetOwnerComputeGraph()->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  const InDataAnchorPtr parent_in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(index));
  GE_CHECK_NOTNULL(parent_in_anchor);
  const OutDataAnchorPtr peer_out_anchor = parent_in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    // Data has and only has one input
    const NodeIndexIO cur_node_info(node, 0, kIn);
    const NodeIndexIO exist_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
    if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != af::GRAPH_SUCCESS) {
      GE_LOGE("[Update][SymbolMapping] failed.");
      return af::GRAPH_FAILED;
    }
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::HandleMergeInput(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                         AnchorToSymbol &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  std::vector<NodeIndexIO> exist_node_infos;
  std::vector<NodeIndexIO> cur_node_infos;
  for (const auto in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      std::string next_name;
      if ((AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, next_name)) && (!next_name.empty())) {
        ComputeGraphPtr graph = node->GetOwnerComputeGraph();
        GE_CHECK_NOTNULL(graph);
        const af::NodePtr next_node = FindNodeFromAllNodes(graph, next_name);
        GE_CHECK_NOTNULL(next_node);
        // NextIteration has and only has one output
        peer_out_anchor = next_node->GetOutDataAnchor(0);
        GE_CHECK_NOTNULL(peer_out_anchor);
        cur_node_infos.emplace_back(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn));
        cur_node_infos.emplace_back(NodeIndexIO(next_node, peer_out_anchor->GetIdx(), kOut));
      }
    } else {
      cur_node_infos.emplace_back(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn));
      exist_node_infos.emplace_back(NodeIndexIO(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut));
    }
  }

  size_t anchor_nums = 0U;
  NodeIndexIO max_node_index_io(static_cast<const Node *>(nullptr), 0, kOut);
  for (const auto &temp_node_info : exist_node_infos) {
    const auto iter1 = anchor_to_symbol.find(temp_node_info.ToString());
    if (iter1 != anchor_to_symbol.end()) {
      const std::string &temp_symbol = iter1->second;
      const auto iter2 = symbol_to_anchors.find(temp_symbol);
      if (iter2 != symbol_to_anchors.end()) {
        if (iter2->second.size() > anchor_nums) {
          max_node_index_io = temp_node_info;
          anchor_nums = iter2->second.size();
        }
      }
    }
  }

  std::string symbol;
  for (const auto &temp_node_info : exist_node_infos) {
    if ((UnionSymbolMapping(max_node_index_io, temp_node_info, symbol_to_anchors, anchor_to_symbol, symbol) !=
         af::GRAPH_SUCCESS) ||
        symbol.empty()) {
      GE_LOGE("[Union][SymbolMap] anchor1:%s & anchor2:%s failed.", max_node_index_io.ToString().c_str(),
              temp_node_info.ToString().c_str());
      return af::GRAPH_FAILED;
    }
  }

  const auto iter = symbol_to_anchors.find(symbol);
  if (iter != symbol_to_anchors.end()) {
    for (const auto &temp_node_info : cur_node_infos) {
      GELOGD("Add anchor %s, symbol %s.", temp_node_info.ToString().c_str(), symbol.c_str());
      iter->second.emplace_back(temp_node_info);
      (void)anchor_to_symbol.emplace(std::make_pair(temp_node_info.ToString(), symbol));
    }
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::HandleSubgraphOutput(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                             AnchorToSymbol &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  const ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(owner_graph);
  const NodePtr parent_node = owner_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  const OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    const OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);

    const auto &in_tensor = op_desc->GetInputDescPtr(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    uint32_t index = 0U;
    if (!af::AttrUtils::GetInt(in_tensor, ATTR_NAME_PARENT_NODE_INDEX, index)) {
      continue;
    }
    GE_CHECK_NOTNULL(parent_node->GetOutDataAnchor(static_cast<int32_t>(index)));
    // Union symbol of peer_out_anchor & parent_out_anchor
    const NodeIndexIO peer_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
    const NodeIndexIO parent_node_info(parent_node, index, kOut);
    std::string symbol;
    if ((UnionSymbolMapping(peer_node_info, parent_node_info, symbol_to_anchors, anchor_to_symbol, symbol) !=
         af::GRAPH_SUCCESS) ||
        symbol.empty()) {
      GE_LOGE("[Union][SymbolMap] anchor1:%s, and anchor2:%s failed.", peer_node_info.ToString().c_str(),
              parent_node_info.ToString().c_str());
      return af::GRAPH_FAILED;
    }

    NodeIndexIO cur_node_info(node, in_data_anchor->GetIdx(), kIn);
    GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
    symbol_to_anchors[symbol].emplace_back(cur_node_info);
    (void)anchor_to_symbol.emplace(std::make_pair(cur_node_info.ToString(), symbol));
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::UnionSymbolMapping(const NodeIndexIO &exist_node_info1, const NodeIndexIO &exist_node_info2,
                                           SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol,
                                           std::string &symbol) {
  const std::string &symbol1 = anchor_to_symbol[exist_node_info1.ToString()];
  const std::string &symbol2 = anchor_to_symbol[exist_node_info2.ToString()];
  if (symbol1 == symbol2) {
    symbol = symbol1;
    GELOGI("no need to union.");
    return af::GRAPH_SUCCESS;
  }

  const auto iter1 = symbol_to_anchors.find(symbol1);
  const auto iter2 = symbol_to_anchors.find(symbol2);
  if ((iter1 == symbol_to_anchors.end()) || (iter2 == symbol_to_anchors.end())) {
    REPORT_INNER_ERR_MSG("E18888", "symbol %s or %s does not exist.", symbol1.c_str(), symbol2.c_str());
    GE_LOGE("[Check][Param] symbol %s or %s does not exist.", symbol1.c_str(), symbol2.c_str());
    return af::GRAPH_FAILED;
  }

  auto &max_iter = ((iter1->second.size() > iter2->second.size()) ? iter1 : iter2);
  auto &min_iter = ((iter1->second.size() > iter2->second.size()) ? iter2 : iter1);
  symbol = ((iter1->second.size() > iter2->second.size()) ? symbol1 : symbol2);
  const std::string min_symbol = ((iter1->second.size() > iter2->second.size()) ? symbol2 : symbol1);
  for (auto &node_index_io : min_iter->second) {
    GELOGD("Update anchor %s, symbol %s.", node_index_io.ToString().c_str(), symbol.c_str());
    max_iter->second.emplace_back(node_index_io);
    const auto iter = anchor_to_symbol.find(node_index_io.ToString());
    GE_ASSERT_TRUE(iter != anchor_to_symbol.end(), "anchor %s does not exist in anchor_to_symbol.",
                   node_index_io.ToString().c_str());
    if (iter->second != min_symbol) {
      GELOGW("[GetRefMapping][Check] not expected symbol of anchor %s, expect %s but got %s.", iter->first.c_str(),
             min_symbol.c_str(), iter->second.c_str());
    }
    iter->second = symbol;
  }

  GELOGI("Union symbol %s and %s succ.", symbol.c_str(), min_symbol.c_str());
  (void)symbol_to_anchors.erase(min_iter);
  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::UpdateRefMapping(const NodeIndexIO &cur_node_info, const NodeIndexIO &exist_node_info,
                                         SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol) {
  const auto iter1 = anchor_to_symbol.find(exist_node_info.ToString());
  if (iter1 == anchor_to_symbol.end()) {
    REPORT_INNER_ERR_MSG("E18888", "data_anchor %s is not visible before data_anchor %s, maybe TopoSorting is missing.",
                         exist_node_info.ToString().c_str(), cur_node_info.ToString().c_str());
    GE_LOGE("[Check][Param] data_anchor %s is not visible before data_anchor %s, maybe TopoSorting is missing.",
            exist_node_info.ToString().c_str(), cur_node_info.ToString().c_str());
    return af::GRAPH_FAILED;
  }

  const std::string &symbol = iter1->second;
  const auto iter2 = symbol_to_anchors.find(symbol);
  if (iter2 == symbol_to_anchors.end()) {
    REPORT_INNER_ERR_MSG("E18888", "symbol %s does not exist in symbol_to_anchors.", symbol.c_str());
    GE_LOGE("[Check][Param] symbol %s not found.", symbol.c_str());
    return af::GRAPH_FAILED;
  }
  GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
  iter2->second.emplace_back(cur_node_info);
  const auto ret = anchor_to_symbol.emplace(std::make_pair(cur_node_info.ToString(), symbol));
  GE_ASSERT_TRUE(ret.first != anchor_to_symbol.end(), "failed to insert anchor to symbol mapping.");
  GE_ASSERT_TRUE(ret.first->second == symbol,
                 "update anchor's symbol failed. cur_node_info: %s, old_symbol: %s, "
                 "new_symbol: %s",
                 cur_node_info.ToString().c_str(), ret.first->second.c_str(), symbol.c_str());
  return af::GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NodePtr GraphUtils::FindNodeFromAllNodes(ComputeGraphPtr &graph,
                                                                                        const std::string &name) {
  const auto root_graph = FindRootGraph(graph);
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param graph is nullptr,check invalid.");
    GE_LOGE("[Check][Param] param graph is nullptr, check invalid");
    return nullptr;
  }

  std::deque<NodePtr> candidates;

  (void)candidates.insert(candidates.begin(), graph->impl_->nodes_.begin(), graph->impl_->nodes_.end());
  while (!candidates.empty()) {
    NodePtr node = candidates.front();
    candidates.pop_front();
    if (node == nullptr) {
      continue;
    }
    if (NodeUtils::IsNameEqual(node, name.c_str())) {
      return node;
    }
    const auto op_desc = node->GetOpDescBarePtr();
    if (op_desc != nullptr) {
      const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
      auto name_iter = subgraph_names.rbegin();
      while (name_iter != subgraph_names.rend()) {
        const auto subgraph = root_graph->GetSubgraph(*name_iter);
        if (subgraph != nullptr) {
          (void)(candidates.insert(candidates.begin(), subgraph->impl_->nodes_.begin(), subgraph->impl_->nodes_.end()));
        }
        ++name_iter;
      }
    }
  }
  return nullptr;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<NodePtr> GraphUtils::FindNodesByTypeFromAllNodes(
    ComputeGraphPtr &graph, const std::string &type) {
  std::vector<NodePtr> nodes;
  const auto &root_graph = FindRootGraph(graph);
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param graph is nullptr,check invalid.");
    GE_LOGE("[Check][Param] param graph is nullptr, check invalid");
    return nodes;
  }

  for (const auto &node : root_graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (NodeUtils::IsTypeEqual(node, type.c_str())) {
      nodes.emplace_back(node);
    }
  }

  return nodes;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<Node *> GraphUtils::FindBareNodesByTypeFromAllNodes(
    ComputeGraphPtr &graph, const char_t *const type) {
  const auto &root_graph = FindRootGraph(graph);
  GE_ASSERT_NOTNULL(root_graph);

  std::vector<Node *> nodes;
  for (const auto node : root_graph->GetAllNodesPtr()) {
    if (strcmp(node->GetTypePtr(), type) == 0) {
      nodes.emplace_back(node);
    }
  }
  return nodes;
}

graphStatus GraphUtils::GetSubgraphsRecursively(const ComputeGraphPtr &graph, std::vector<ComputeGraphPtr> &subgraphs) {
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to find root graph");
    GELOGE(af::GRAPH_FAILED, "[Get][Graph] Failed to find root graph");
    return af::GRAPH_FAILED;
  }
  if (graph == root_graph) {
    subgraphs = graph->GetAllSubgraphs();
    return af::GRAPH_SUCCESS;
  }
  for (const auto &node : graph->GetAllNodes()) {
    // op_desc of node should not be null
    for (const auto &graph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
      const auto &subgraph = root_graph->GetSubgraph(graph_name);
      if (subgraph == nullptr) {
        GELOGW("[Get][Subgraph] subgraph %s of node %s is null", graph_name.c_str(), node->GetName().c_str());
        continue;
      }
      subgraphs.emplace_back(subgraph);
    }
  }
  return af::GRAPH_SUCCESS;
}

/// Check if out_data_anchor is reference of input
/// @param [in] out_data_anchor
/// @param [out] reuse_in_index
/// @return bool
bool GraphUtils::IsRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index) {
  if (out_data_anchor == nullptr) {
    GELOGW("[Check][Param] out_data_anchor is null");
    return false;
  }
  const int32_t output_index = out_data_anchor->GetIdx();

  // pass-through op
  const auto node = out_data_anchor->GetOwnerNodeBarePtr();
  const std::string &type = node->GetType();
  static const std::unordered_set<std::string> pass_through_types = {NETOUTPUT, WHILE, _WHILE, STATELESSWHILE};
  if ((pass_through_types.count(type) > 0U) || (NodeUtils::IsSubgraphInput(node))) {
    reuse_in_index = output_index;
    GELOGI("Pass-Through node name[%s] index[%u].", node->GetName().c_str(), reuse_in_index);
    return true;
  }

  // Merge op 0th output
  const bool is_ge_local_op = ((type == MERGE) || (type == RESHAPE)) && (output_index == 0);
  if (is_ge_local_op) {
    reuse_in_index = 0;
    GELOGI("%s name[%s] output_index[0] reuse input 0.", type.c_str(), node->GetName().c_str());
    return true;
  }

  // ref op
  // op_desc of node should not be null
  const auto op_desc = node->GetOpDescBarePtr();
  bool is_ref = false;
  (void)af::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (is_ref) {
    const std::string &output_name = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(output_index));
    for (const auto &input_name : op_desc->GetAllInputNames()) {
      if ((!input_name.empty()) && (output_name == input_name)) {
        reuse_in_index = op_desc->GetInputIndexByName(input_name);
        GELOGD("Reference name[%s] output[%s][%d] ref to input[%s][%d].", op_desc->GetName().c_str(),
               output_name.c_str(), output_index, input_name.c_str(), reuse_in_index);
        return true;
      }
    }
  }

  // reuse input
  const auto output_op_desc = op_desc->GetOutputDescPtr(static_cast<uint32_t>(output_index));
  if (output_op_desc != nullptr) {
    bool reuse_input = false;
    if ((TensorUtils::GetReuseInput(*output_op_desc, reuse_input) == af::GRAPH_SUCCESS) && reuse_input) {
      uint32_t reuse_input_index = 0U;
      if (TensorUtils::GetReuseInputIndex(*output_op_desc, reuse_input_index) == af::GRAPH_SUCCESS) {
        reuse_in_index = static_cast<int32_t>(reuse_input_index);
        GELOGI("ReuseInput name[%s] output[%d] reuse input[%d].", op_desc->GetName().c_str(), output_index,
               reuse_in_index);
        return true;
      }
    }
  }
  // nopadding reuse
  return IsNoPaddingRefFromInput(out_data_anchor, reuse_in_index);
}

graphStatus GraphUtils::CheckIsRefFromOther(const OutDataAnchorPtr &out_data_anchor, NodePtr &refed_node,
                                            bool &is_ref_from_other) {
  GE_ASSERT_NOTNULL(out_data_anchor);
  const auto owner_node = out_data_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(owner_node);
  bool is_ref_from_refdata = false;
  bool is_ref_from_innerdata = false;
  GE_ASSERT_GRAPH_SUCCESS(CheckIsRefFromRefData(out_data_anchor, refed_node, is_ref_from_refdata));
  GE_ASSERT_GRAPH_SUCCESS(CheckIsRefFromInnerData(out_data_anchor, refed_node, is_ref_from_innerdata));
  is_ref_from_other = (is_ref_from_refdata || is_ref_from_innerdata);
  return af::GRAPH_SUCCESS;
}

bool GraphUtils::IsNoPaddingRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index) {
  const auto node = out_data_anchor->GetOwnerNodeBarePtr();
  // nopadding means output[0] reuse input[0], but as history reason,
  // other output index also return true for mem assign in block_mem_assigner
  bool attr_reuse = false;
  bool is_input_continuous = false;
  bool is_out_continuous = false;
  (void)af::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_input_continuous);
  (void)af::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_out_continuous);
  const bool get_reuse_flag =
      af::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
  const bool is_no_padding_reuse_input = (is_input_continuous || is_out_continuous) && get_reuse_flag && attr_reuse;
  if (is_no_padding_reuse_input) {
    reuse_in_index = 0;
    GELOGI("Nopadding ReuseInput name[%s] output[%d] reuse input[%d].", node->GetName().c_str(),
           out_data_anchor->GetIdx(), reuse_in_index);
    return true;
  }
  return false;
}

bool GraphUtils::IsNodeInGraphRecursively(const ComputeGraphPtr &graph, const Node &node) {
  auto parent_graph = node.GetOwnerComputeGraph();
  while (parent_graph != nullptr) {
    if (parent_graph == graph) {
      return true;
    }
    parent_graph = parent_graph->GetParentGraph();
  }
  return false;
}

/// Determine if the graph is a UNKNOWN_SHAPE graph based on whether the graph and all subgraphs
/// of the graph have UNKNOWN_SHAPE operators or not.
/// Note: This function will only look 'down' from the graph, not 'up'. For example, the following
/// scenario (K for known shape, U for unknown shape), ROOT graph is UNKNOWN_SHAPE while SUB graph is KNOWN_SHAPE
/// ROOT graph:      A -----> B -----> C
///                  K    subgraph     U
///                           |
///                           V
/// SUB graph:          D --> E --> F
///                     K     K     K
/// @param [in] graph
/// @return bool
bool GraphUtils::IsUnknownShapeGraph(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    GELOGW("[Check][Param] Input graph is nullptr.");
    return false;
  }
  for (const auto &node : graph->GetDirectNode()) {
    bool is_unknown = false;
    const auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
    if (ret != af::GRAPH_SUCCESS) {
      GELOGW("[Check][UnknownGraph] Get unknown status failed, node name:%s, type:%s", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    if (is_unknown) {
      GELOGD("Node %s, type %s is unknown shape in graph %s.", node->GetName().c_str(), node->GetType().c_str(),
             graph->GetName().c_str());
      return true;
    }
  }
  GELOGD("Graph %s does not have unknown shape node.", graph->GetName().c_str());
  return false;
}

ComputeGraphPtr GraphUtils::BuildSubgraphWithNodes(const ComputeGraphPtr &graph, const std::set<NodePtr> &nodes,
                                                   const std::string &subgraph_name) {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Graph is null");
    GELOGE(FAILED, "[Check][Param] graph is null");
    return nullptr;
  }
  return BuildSubgraphWithNodes(*graph, nodes, subgraph_name);
}

ComputeGraphPtr GraphUtils::BuildSubgraphWithNodes(ComputeGraph &graph, const std::set<NodePtr> &nodes,
                                                   const std::string &subgraph_name) {
  if (nodes.empty()) {
    GELOGW("nodes are empty, no need to build subgraph");
    return nullptr;
  }

  GraphInfo graph_info;
  BuildGraphInfoFromNodes(nodes, graph_info);

  const NodePtr graph_node = BuildSubgraphNode(graph, subgraph_name, graph_info);
  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Build SubgraphNode failed, subgraph_name:%s.", subgraph_name.c_str());
    GELOGE(FAILED, "[Build][SubgraphNode] failed, subgraph_name:%s.", subgraph_name.c_str());
    return nullptr;
  }

  const ComputeGraphPtr subgraph = BuildSubgraph(graph_node, graph_info, subgraph_name);
  if (subgraph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Build Subgraph %s failed", subgraph_name.c_str());
    GELOGE(FAILED, "[Build][Subgraph] %s failed", subgraph_name.c_str());
    return nullptr;
  }
  const auto &root_graph = GraphUtils::FindRootGraph(graph_node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Find root graph failed, graph:%s", graph.GetName().c_str());
    GELOGE(FAILED, "[Find][RootGraph] failed, graph:%s", graph.GetName().c_str());
    return nullptr;
  }
  if (root_graph->AddSubgraph(subgraph) != af::GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Add subgraph %s failed, root graph:%s", subgraph->GetName().c_str(),
                         root_graph->GetName().c_str());
    GELOGE(FAILED, "[Add][SubGraph] %s failed, root graph:%s", subgraph->GetName().c_str(),
           root_graph->GetName().c_str());
    return nullptr;
  }

  if ((RelinkDataEdges(graph_node, graph_info) != af::GRAPH_SUCCESS) ||
      (RelinkCtrlEdges(graph_node, graph_info) != af::GRAPH_SUCCESS)) {
    REPORT_INNER_ERR_MSG("E18888", "ReLink edges for graph %s failed, graph_node:%s", graph.GetName().c_str(),
                         graph_node->GetName().c_str());
    GELOGE(FAILED, "[ReLink][Edges] for graph %s failed, graph_node:%s", graph.GetName().c_str(),
           graph_node->GetName().c_str());
    return nullptr;
  }

  for (const auto &node : nodes) {
    // op_desc of node should not be null
    const auto subgraph_names_inner = node->GetOpDesc()->GetSubgraphInstanceNames();
    for (const auto &subgraph_name_inner : subgraph_names_inner) {
      node->GetOpDesc()->RemoveSubgraphInstanceName(subgraph_name_inner);
    }
    if (RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node) != af::GRAPH_SUCCESS) {
      GELOGW("Remove node %s failed.", node->GetName().c_str());
    }
  }

  return subgraph;
}
template <typename Container>
void GraphUtils::BuildGraphInfoFromNodes(const Container &nodes, GraphInfo &graph_info) {
  // 将 nodes 转换为有序容器（std::vector），并按节点的唯一标识符排序
  std::vector<NodePtr> ordered_nodes(nodes.begin(), nodes.end());

  // 按节点的唯一标识符排序
  std::sort(ordered_nodes.begin(), ordered_nodes.end(),
            [](const NodePtr &a, const NodePtr &b) { return a->GetName() < b->GetName(); });

  std::map<OutDataAnchorPtr, size_t> data_input_index_map;
  for (const auto &node : ordered_nodes) {
    // graph nodes
    (void)graph_info.nodes_.emplace(node);
    // in data
    BuildInDataEdgesFromNode(node, nodes, data_input_index_map, graph_info);
    // out data
    std::list<InDataAnchorPtr> peer_data_anchors;
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      peer_data_anchors.clear();
      const auto &peer_in_anchors = out_data_anchor->GetPeerInDataAnchors();
      (void)std::copy_if(peer_in_anchors.begin(), peer_in_anchors.end(), std::back_inserter(peer_data_anchors),
                         [nodes](const InDataAnchorPtr &peer_in_anchor) {
                           return nodes.count(peer_in_anchor->GetOwnerNode()) == 0UL;
                         });
      if (!peer_data_anchors.empty()) {
        const size_t output_index = graph_info.data_outputs_.size();
        graph_info.data_outputs_[output_index] = std::make_pair(out_data_anchor, peer_data_anchors);
      }
    }
    // in ctrl
    for (const auto &in_ctrl_node : node->GetInControlNodes()) {
      if (nodes.count(in_ctrl_node) == 0UL) {
        graph_info.ctrl_inputs_.emplace_back(in_ctrl_node->GetOutControlAnchor(), node->GetInControlAnchor());
      } else {
        graph_info.inner_ctrl_edges_.emplace_back(
            std::make_pair(in_ctrl_node->GetOutControlAnchor(), node->GetInControlAnchor()));
      }
    }
    // out ctrl
    for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (nodes.count(out_ctrl_node) == 0UL) {
        graph_info.ctrl_outputs_.emplace_back(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor());
      }
    }
  }
}
template <typename Container>
void GraphUtils::BuildInDataEdgesFromNode(const NodePtr &node, const Container &nodes,
                                          std::map<OutDataAnchorPtr, size_t> &data_input_index_map,
                                          GraphInfo &graph_info) {
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (nodes.count(peer_out_anchor->GetOwnerNode()) == 0UL) {
      size_t input_index;
      if (data_input_index_map.count(peer_out_anchor) == 0UL) {
        input_index = graph_info.data_inputs_.size();
        data_input_index_map[peer_out_anchor] = input_index;
        graph_info.data_inputs_[input_index].first = peer_out_anchor;
      } else {
        input_index = data_input_index_map[peer_out_anchor];
      }
      graph_info.data_inputs_[input_index].second.emplace_back(in_data_anchor);
    } else {
      graph_info.inner_data_edges_.emplace_back(std::make_pair(peer_out_anchor, in_data_anchor));
    }
  }
}

NodePtr GraphUtils::BuildSubgraphNode(ComputeGraph &graph, const std::string &graph_name, const GraphInfo &graph_info) {
  OpDescBuilder op_desc_builder(graph_name + "_" + PARTITIONEDCALL, PARTITIONEDCALL);
  int32_t i = 0;
  for (const auto &item : graph_info.data_inputs_) {
    for (const auto &in_data_anchor : item.second.second) {
      const auto input_desc = in_data_anchor->GetOwnerNodeBarePtr()->GetOpDesc();
      if (input_desc == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "op_desc is null, node:%s", in_data_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(PARAM_INVALID, "[Check][Param] op_desc is null, node:%s",
               in_data_anchor->GetOwnerNode()->GetName().c_str());
        return nullptr;
      }
      (void)op_desc_builder.AddInput("args" + std::to_string(i),
                                     input_desc->GetInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx())));
      i++;
    }
  }
  for (const auto &item : graph_info.data_outputs_) {
    const auto output_desc = item.second.first->GetOwnerNodeBarePtr()->GetOpDesc();
    if (output_desc == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "op_desc is null, node:%s", item.second.first->GetOwnerNode()->GetName().c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] op_desc is null, node:%s",
             item.second.first->GetOwnerNode()->GetName().c_str());
      return nullptr;
    }
    (void)op_desc_builder.AddOutput("output" + std::to_string(item.first),
                                    output_desc->GetOutputDesc(static_cast<uint32_t>(item.second.first->GetIdx())));
  }

  const OpDescPtr op_desc = op_desc_builder.Build();
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Create op_desc for subgraph node failed, name:%s.", graph_name.c_str());
    GELOGE(FAILED, "[Create][OpDesc] for subgraph node failed, name:%s.", graph_name.c_str());
    return nullptr;
  }

  (void)op_desc->AddSubgraphName("f");
  (void)op_desc->SetSubgraphInstanceName(0U, graph_name);

  return graph.AddNode(op_desc);
}

ComputeGraphPtr GraphUtils::BuildGraph(const GraphInfo &graph_info, const std::string &name) {
  return BuildGraphInternal(graph_info, name, nullptr);  // 普通图，parent_node 为 nullptr
}

ComputeGraphPtr GraphUtils::BuildSubgraph(const NodePtr &subgraph_node, const GraphInfo &graph_info,
                                          const std::string &subgraph_name) {
  return BuildGraphInternal(graph_info, subgraph_name, subgraph_node);  // 子图，传入 parent_node
}

graphStatus GraphUtils::RelinkDataEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info) {
  // in data nodes
  int32_t i = 0;
  for (const auto &item : graph_info.data_inputs_) {
    for (const auto &in_data_anchor : item.second.second) {
      GE_CHK_STATUS_RET(item.second.first->Unlink(in_data_anchor), "[Remove][DataEdge] %s:%d->%s:%d failed",
                        item.second.first->GetOwnerNode()->GetName().c_str(), item.second.first->GetIdx(),
                        in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());
      GE_CHK_STATUS_RET(item.second.first->LinkTo(subgraph_node->GetInDataAnchor(i)),
                        "[Add][DataEdge] %s:%d->%s:%u failed.", item.second.first->GetOwnerNode()->GetName().c_str(),
                        item.second.first->GetIdx(), subgraph_node->GetName().c_str(), item.first);
      i++;
    }
  }
  // out data nodes
  for (const auto &item : graph_info.data_outputs_) {
    const auto &out_data_anchor = subgraph_node->GetOutDataAnchor(static_cast<int32_t>(item.first));
    GE_CHECK_NOTNULL(out_data_anchor);
    for (const auto &peer_in_anchor : item.second.second) {
      GE_CHK_STATUS_RET(item.second.first->Unlink(peer_in_anchor), "[Remove][DataEdge] %s:%d->%s:%d failed.",
                        item.second.first->GetOwnerNode()->GetName().c_str(), item.second.first->GetIdx(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
      GE_CHK_STATUS_RET(out_data_anchor->LinkTo(peer_in_anchor), "[Add][DataEdge] %s:%u->%s:%d failed.",
                        subgraph_node->GetName().c_str(), item.first, peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetIdx());
    }
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::RelinkCtrlEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info) {
  // in ctrl nodes
  for (const auto &ctrl_input : graph_info.ctrl_inputs_) {
    GE_CHK_STATUS_RET(ctrl_input.first->Unlink(ctrl_input.second), "[Remove][CtrlEdge] %s->%s failed",
                      ctrl_input.first->GetOwnerNode()->GetName().c_str(),
                      ctrl_input.second->GetOwnerNode()->GetName().c_str());
    if (!ctrl_input.first->IsLinkedWith(subgraph_node->GetInControlAnchor())) {
      GE_CHK_STATUS_RET(ctrl_input.first->LinkTo(subgraph_node->GetInControlAnchor()), "[Add][CtrlEdge] %s->%s failed.",
                        ctrl_input.first->GetOwnerNode()->GetName().c_str(), subgraph_node->GetName().c_str());
    }
  }
  // out ctrl nodes
  for (const auto &ctrl_output : graph_info.ctrl_outputs_) {
    GE_CHK_STATUS_RET(ctrl_output.first->Unlink(ctrl_output.second), "[Remove][CtrlEdge] %s->%s failed.",
                      ctrl_output.first->GetOwnerNode()->GetName().c_str(),
                      ctrl_output.second->GetOwnerNode()->GetName().c_str());
    if (!subgraph_node->GetOutControlAnchor()->IsLinkedWith(ctrl_output.second)) {
      GE_CHK_STATUS_RET(subgraph_node->GetOutControlAnchor()->LinkTo(ctrl_output.second),
                        "[Add][CtrlEdge] %s->%s failed.", subgraph_node->GetName().c_str(),
                        ctrl_output.second->GetOwnerNode()->GetName().c_str());
    }
  }

  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::UnfoldSubgraph(const ComputeGraphPtr &graph,
                                       const std::function<bool(const ComputeGraphPtr &)> &filter) {
  GE_CHECK_NOTNULL(graph);
  const auto &parent_graph = graph->GetParentGraph();
  const auto &parent_node = graph->GetParentNode();
  if ((parent_graph == nullptr) && (parent_node == nullptr)) {
    return af::GRAPH_SUCCESS;
  }

  return UnfoldGraph(graph, parent_graph, parent_node, filter);
}

graphStatus GraphUtils::UnfoldGraph(const ComputeGraphPtr &graph, const ComputeGraphPtr &target_graph,
                                    const NodePtr &target_node, const function<bool(const ComputeGraphPtr &)> &filter,
                                    int32_t depth) {
  if (depth >= kCopyGraphMaxRecursionDepth) {
    REPORT_INNER_ERR_MSG("E18888", "param depth:%d >= %d(allow max subgraphs)", depth, kCopyGraphMaxRecursionDepth);
    GELOGE(af::GRAPH_FAILED, "[Check][Param] too many subgraphs exist: %d > %d (allow max subgraphs)", depth,
           kCopyGraphMaxRecursionDepth);
    return af::GRAPH_FAILED;
  }
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(target_graph);
  GE_CHECK_NOTNULL(target_node);

  GE_CHK_STATUS_RET(MergeInputNodes(graph, target_node),
                    "[Invoke][MergeInputNodes] Merge data nodes for graph %s failed", graph->GetName().c_str());
  GE_CHK_STATUS_RET(MergeNetOutputNode(graph, target_node),
                    "[Invoke][MergeNetOutputNode] Merge net output nodes for graph %s failed",
                    graph->GetName().c_str());
  GELOGD("[%s] Merging graph inputs and outputs successfully", graph->GetName().c_str());

  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if ((node->GetType() == DATA) || (node->GetType() == NETOUTPUT)) {
      continue;
    }

    std::vector<ComputeGraphPtr> subgraphs;
    GE_CHK_STATUS_RET(NodeUtils::GetDirectSubgraphs(node, subgraphs), "[Get][Subgraphs] failed, graph:%s",
                      node->GetName().c_str());
    bool skip_add_node_flag = true;
    for (const auto &subgraph : subgraphs) {
      if ((filter != nullptr) && filter(subgraph)) {
        GE_CHK_STATUS_RET(
            UnfoldGraph(subgraph, subgraph->GetParentGraph(), subgraph->GetParentNode(), filter, depth + 1),
            "[Invoke][UnfoldSubgraph] Failed to merge graph %s", subgraph->GetName().c_str());
        skip_add_node_flag = false;
      } else {
        subgraph->SetParentGraph(target_graph);
      }
    }

    if (skip_add_node_flag) {
      (void)target_graph->AddNode(node);
      GELOGD("[%s::%s] added to target graph: [%s].", graph->GetName().c_str(), node->GetName().c_str(),
             target_graph->GetName().c_str());
      (void)node->SetOwnerComputeGraph(target_graph);
    }
  }

  GELOGD("[%s] Done merging graph. remove it from root graph", graph->GetName().c_str());

  const auto &subgraph_name = graph->GetName();
  const auto &root_graph = GraphUtils::FindRootGraph(target_graph);
  GE_CHECK_NOTNULL(root_graph);
  root_graph->RemoveSubgraph(graph->GetName());
  target_node->GetOpDesc()->RemoveSubgraphInstanceName(subgraph_name);
  if (RemoveNodeWithoutRelink(target_graph, target_node) != af::GRAPH_SUCCESS) {
    GELOGW("Remove node %s failed, graph:%s.", target_node->GetName().c_str(), target_graph->GetName().c_str());
  }

  return SUCCESS;
}

graphStatus GraphUtils::MergeInputNodes(const ComputeGraphPtr &graph, const NodePtr &target_node) {
  GE_CHECK_NOTNULL(target_node);

  std::set<NodePtr> src_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != DATA) {
      if (node->GetInAllNodes().empty()) {
        (void)src_nodes.emplace(node);
      }
      continue;
    }

    uint32_t parent_index = 0U;
    if ((!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) &&
        (!AttrUtils::GetInt(node->GetOpDesc(), "index", parent_index))) {
      REPORT_INNER_ERR_MSG("E18888", "Get attr {%s} or {index} failed, node:%s", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                           node->GetName().c_str());
      GELOGE(FAILED, "[Get][Attr] {%s} or {index} failed, node:%s", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
             node->GetName().c_str());
      return af::GRAPH_FAILED;
    }

    const auto parent_node_in_anchor = target_node->GetInDataAnchor(static_cast<int32_t>(parent_index));
    GE_CHECK_NOTNULL(parent_node_in_anchor);
    const auto src_out_anchor = parent_node_in_anchor->GetPeerOutAnchor();
    if ((src_out_anchor == nullptr) || (src_out_anchor->GetOwnerNodeBarePtr() == nullptr)) {
      continue;
    }
    parent_node_in_anchor->UnlinkAll();

    // link src to outputs of DataNode
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto dst_node = peer_in_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(dst_node);
        const auto &in_nodes = dst_node->GetInDataNodes();
        if (std::all_of(in_nodes.begin(), in_nodes.end(), [](const NodePtr &n) { return n->GetType() == DATA; })) {
          (void)src_nodes.emplace(dst_node);
        }
        GE_CHK_STATUS_RET(ReplaceEdgeSrc(out_data_anchor, peer_in_anchor, src_out_anchor),
                          "[Replace][DataEdge] %s:%d->%s:%d failed", out_data_anchor->GetOwnerNode()->GetNamePtr(),
                          out_data_anchor->GetIdx(), peer_in_anchor->GetOwnerNode()->GetNamePtr(),
                          peer_in_anchor->GetIdx());
      }
    }
    // when unfold partitonCall, if data have control edges, which will be left in final graph
    // which cause topo sort failed.
    auto out_control_anchor = node->GetOutControlAnchor();
    GE_CHECK_NOTNULL(out_control_anchor);
    out_control_anchor->UnlinkAll();
  }

  // transfer in control edges to all root nodes
  for (const auto &src_node : src_nodes) {
    const auto &in_nodes = src_node->GetInAllNodes();
    const std::set<NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (const auto &in_control_node : target_node->GetInControlNodes()) {
      GE_CHECK_NOTNULL(in_control_node);
      if ((in_node_set.count(in_control_node) == 0UL) && (kMergeInputSkipTypes.count(src_node->GetType()) == 0UL)) {
        GELOGD("[%s] Restore control edge to [%s]", in_control_node->GetName().c_str(), src_node->GetName().c_str());
        (void)AddEdge(in_control_node->GetOutControlAnchor(), src_node->GetInControlAnchor());
      }
    }
  }

  target_node->GetInControlAnchor()->UnlinkAll();
  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::MergeNetOutputNode(const ComputeGraphPtr &graph, const NodePtr &target_node) {
  GE_CHECK_NOTNULL(target_node);

  const NodePtr &net_output = graph->FindFirstNodeMatchType(NETOUTPUT);
  if (net_output == nullptr) {
    GELOGD("Graph has no NetOutput node, no need to merge");
    return SUCCESS;
  }
  auto all_in_nodes = net_output->GetInAllNodes();
  auto all_out_nodes = target_node->GetOutAllNodes();
  net_output->GetInControlAnchor()->UnlinkAll();
  target_node->GetOutControlAnchor()->UnlinkAll();

  for (const auto &in_data_anchor : net_output->GetAllInDataAnchors()) {
    GE_CHECK_NOTNULL(in_data_anchor);
    const auto index = in_data_anchor->GetIdx();
    uint32_t parent_index = index;
    // op_desc of node should not be null
    if (!AttrUtils::GetInt(net_output->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(index)),
                           ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGW("SubGraph: %s NetOutput input tensor %d, attr %s not found, use anchor index %u.",
             graph->GetName().c_str(), index, ATTR_NAME_PARENT_NODE_INDEX.c_str(), parent_index);
    }

    const auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNodeBarePtr());
    GE_CHK_STATUS_RET(RemoveEdge(src_out_anchor, in_data_anchor), "[Remove][DataEdge] %s:%d->%s:%d failed",
                      src_out_anchor->GetOwnerNode()->GetName().c_str(), src_out_anchor->GetIdx(),
                      net_output->GetName().c_str(), in_data_anchor->GetIdx());

    const OutDataAnchorPtr &parent_out_anchor = target_node->GetOutDataAnchor(static_cast<int32_t>(parent_index));
    GE_CHECK_NOTNULL(parent_out_anchor);
    for (InDataAnchorPtr &dst_in_anchor : parent_out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS_RET(ReplaceEdgeSrc(parent_out_anchor, dst_in_anchor, src_out_anchor),
                        "[Replace][DataEdge] %s:%d->%s:%d failed", parent_out_anchor->GetOwnerNode()->GetNamePtr(),
                        parent_out_anchor->GetIdx(), dst_in_anchor->GetOwnerNode()->GetNamePtr(),
                        dst_in_anchor->GetIdx());
    }
  }

  // transfer out control edges
  const OrderedNodeSet in_node_set(all_in_nodes.begin(), all_in_nodes.end());
  const OrderedNodeSet out_node_set(all_out_nodes.begin(), all_out_nodes.end());
  for (auto &src_node : in_node_set) {
    GELOGD("[%s] process in node.", src_node->GetName().c_str());
    auto out_nodes = src_node->GetOutAllNodes();
    const std::set<NodePtr> node_set(out_nodes.begin(), out_nodes.end());
    for (auto &dst_node : out_node_set) {
      if (node_set.count(dst_node) == 0UL) {
        GELOGD("[%s] Restore control edge to [%s]", src_node->GetName().c_str(), dst_node->GetName().c_str());
        (void)src_node->GetOutControlAnchor()->LinkTo(dst_node->GetInControlAnchor());
      }
    }
  }

  return af::GRAPH_SUCCESS;
}

void GraphUtils::InheritOriginalAttr(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph) {
  const std::map<string, GeAttrValue> &original_attrs = AttrUtils::GetAllAttrs(src_compute_graph);
  for (auto const &attr_iter : original_attrs) {
    if (dst_compute_graph->TrySetAttr(attr_iter.first, attr_iter.second) != af::GRAPH_SUCCESS) {
      GELOGW("Set inherit original attr[%s] failed, Please Check.", attr_iter.first.c_str());
    }
  }
}

bool GraphUtils::IsSingleOpScene(const ComputeGraphPtr &graph) {
  bool is_single_op = false;
  if (AttrUtils::GetBool(graph, ATTR_SINGLE_OP_SCENE, is_single_op)) {
    return is_single_op;
  }
  GELOGD("There is no _single_op_scene for graph:%s. Start search all node.", graph->GetName().c_str());
  for (const auto &node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    if (AttrUtils::GetBool(node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, is_single_op)) {
      return is_single_op;
    }
  }
  return is_single_op;
}

CycleDetectorPtr GraphUtils::CreateCycleDetector(const ComputeGraphPtr &graph) {
  CycleDetectorPtr detector = ComGraphMakeUnique<CycleDetector>();
  if (detector == nullptr) {
    GELOGW("Fail to create cycle detector. Return null.");
    return nullptr;
  }
  const auto ret = detector->Init(graph);
  if (ret != SUCCESS) {
    GELOGW("Fail to init cycle detector. Return null.");
    return nullptr;
  }
  return detector;
}

CycleDetectorSharedPtr GraphUtils::CreateSharedCycleDetector(const ComputeGraphPtr &graph) {
  CycleDetectorSharedPtr detector = nullptr;
  GE_MAKE_SHARED(detector = std::make_shared<CycleDetector>(), return nullptr);
  if (detector == nullptr) {
    GELOGW("Fail to create cycle detector. Return null.");
    return nullptr;
  }
  const auto ret = detector->Init(graph);
  if (ret != SUCCESS) {
    GELOGW("Fail to init cycle detector. Return null.");
    return nullptr;
  }
  return detector;
}

/// @brief Add node to graph
/// @param [in] op_desc
/// @return ComputeGraphBuilder
ComputeGraphBuilder &ComputeGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  nodes_.emplace_back(op_desc);
  return *this;
}

/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return ComputeGraphBuilder
ComputeGraphBuilder &ComputeGraphBuilder::AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                                      const std::string &dst_name, const uint32_t in_anchor_ind) {
  data_links_.emplace_back(
      std::make_pair(std::make_pair(src_name, out_anchor_ind), std::make_pair(dst_name, in_anchor_ind)));
  return *this;
}

/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return ComputeGraphBuilder
ComputeGraphBuilder &ComputeGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  ctrl_links_.emplace_back(std::make_pair(src_name, dst_name));
  return *this;
}

/// @brief Build nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void ComputeGraphBuilder::BuildNodes(graphStatus &error_code, std::string &error_msg) {
  if (owner_graph_ == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return;
  }

  std::string node_name;
  for (auto &op_desc : nodes_) {
    if (op_desc == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "op_desc is NULL.";
      return;
    }

    node_name = op_desc->GetName();
    const NodePtr node = owner_graph_->AddNode(op_desc);
    if (node == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "Add node " + node_name + " failed.";
      return;
    }

    GELOGD("Add node name:%s, type:%s.", node_name.c_str(), op_desc->GetType().c_str());
    node_names_[node_name] = node;
  }

  GELOGD("BuildNodes succ.");
}

/// @brief Build data-links
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void ComputeGraphBuilder::BuildDataLinks(graphStatus &error_code, std::string &error_msg) {
  for (auto &pair : data_links_) {
    const std::string src_name = pair.first.first;
    const auto out_ind = static_cast<int32_t>(pair.first.second);
    const std::string dst_name = pair.second.first;
    const auto in_ind = static_cast<int32_t>(pair.second.second);
    std::string log_msg = "Add data-edge ";
    (void)log_msg.append(src_name)
        .append(":")
        .append(std::to_string(out_ind))
        .append("->")
        .append(dst_name)
        .append(":")
        .append(std::to_string(in_ind));

    const auto src_iter = node_names_.find(src_name);
    const auto dst_iter = node_names_.find(dst_name);
    if ((src_iter == node_names_.end()) || (dst_iter == node_names_.end())) {
      error_code = af::GRAPH_FAILED;
      error_msg = log_msg + " failed: node does not exist in graph.";
      return;
    }

    const NodePtr src_node = node_names_[src_name];
    const NodePtr dst_node = node_names_[dst_name];
    if ((src_node == nullptr) || (dst_node == nullptr)) {
      error_code = af::GRAPH_FAILED;
      error_msg = log_msg + " failed: node is NULL.";
      return;
    }

    if (GraphUtils::AddEdge(src_node->GetOutDataAnchor(out_ind), dst_node->GetInDataAnchor(in_ind)) !=
        af::GRAPH_SUCCESS) {
      error_code = af::GRAPH_FAILED;
      error_msg = log_msg + " failed.";
      return;
    }

    GELOGD("%s succ.", log_msg.c_str());
  }

  GELOGD("BuildDataLinks succ.");
}

/// @brief Build ctrl-links
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void ComputeGraphBuilder::BuildCtrlLinks(graphStatus &error_code, std::string &error_msg) {
  for (auto &pair : ctrl_links_) {
    const std::string src_name = pair.first;
    const std::string dst_name = pair.second;
    std::string log_msg = "Add ctrl-edge ";
    (void)log_msg.append(src_name).append("->").append(dst_name);

    const auto src_iter = node_names_.find(src_name);
    const auto dst_iter = node_names_.find(dst_name);
    if ((src_iter == node_names_.end()) || (dst_iter == node_names_.end())) {
      error_code = af::GRAPH_FAILED;
      error_msg = log_msg + " failed: node does not exist in graph.";
      return;
    }

    const NodePtr src_node = node_names_[src_name];
    const NodePtr dst_node = node_names_[dst_name];
    if ((src_node == nullptr) || (dst_node == nullptr)) {
      error_code = af::GRAPH_FAILED;
      error_msg = log_msg + " failed: node is NULL.";
      return;
    }

    if (GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor()) != af::GRAPH_SUCCESS) {
      error_code = af::GRAPH_FAILED;
      error_msg = log_msg + " failed.";
      return;
    }

    GELOGD("%s succ.", log_msg.c_str());
  }

  GELOGD("BuildCtrlLinks succ.");
}

/// @brief Get node with name
/// @param [in] name
/// @return NodePtr
NodePtr ComputeGraphBuilder::GetNode(const std::string &name) {
  const auto iter = node_names_.find(name);
  if (iter == node_names_.end()) {
    REPORT_INNER_ERR_MSG("E18888", "node %s does not exist.", name.c_str());
    GE_LOGE("[Check][Param] node %s does not exist.", name.c_str());
    return nullptr;
  }
  return iter->second;
}

/// @brief Get all nodes
/// @return std::vector<NodePtr>
std::vector<NodePtr> ComputeGraphBuilder::GetAllNodes() {
  std::vector<NodePtr> nodes;
  for (const auto &iter : node_names_) {
    nodes.emplace_back(iter.second);
  }
  return nodes;
}

/// @brief Add node to graph
/// @param [in] op_desc
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  (void)ComputeGraphBuilder::AddNode(op_desc);
  return *this;
}

/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                                        const std::string &dst_name, const uint32_t in_anchor_ind) {
  (void)ComputeGraphBuilder::AddDataLink(src_name, out_anchor_ind, dst_name, in_anchor_ind);
  return *this;
}

/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  (void)ComputeGraphBuilder::AddControlLink(src_name, dst_name);
  return *this;
}

/// @brief Set index_th input anchor for graph
/// @param [in] index
/// @param [in] node_names
/// @param [in] anchor_inds
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::SetInput(const uint32_t index, const std::vector<std::string> &node_names,
                                                     const std::vector<uint32_t> &anchor_inds) {
  graph_inputs_[index] = std::make_pair(node_names, anchor_inds);
  return *this;
}

/// @brief Set index_th input of graph as useless
/// @param [in] index
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::SetUselessInput(const uint32_t index) {
  graph_inputs_[index] = std::make_pair(std::vector<std::string>(), std::vector<uint32_t>());
  return *this;
}

/// @brief Add output anchor for graph
/// @param [in] owner_node_name
/// @param [in] anchor_ind
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::AddOutput(const std::string &owner_node_name, uint32_t anchor_ind) {
  graph_outputs_.emplace_back(std::make_pair(owner_node_name, anchor_ind));
  return *this;
}

/// @brief Add target for graph
/// @param [in] target_name
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::AddTarget(const std::string &target_name) {
  graph_targets_.emplace_back(target_name);
  return *this;
}

/// @brief Set parent-node of graph
/// @param [in] parent_node
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::SetParentNode(const NodePtr &parent_node) {
  parent_node_ = parent_node;
  return *this;
}

/// @brief Set mapping-relation of parent-node in_anchor_ind & Data-node
/// @param [in] input_mapping: index_of_graph_input -> in_anchor_index_of_parent_node
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::SetInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  for (auto &item : input_mapping) {
    input_mapping_[item.first] = item.second;
  }
  return *this;
}

/// @brief Set mapping-relation of parent-node out_anchor_ind & NetOutput-node out_anchor_ind
/// @param [in] output_mapping: index_of_graph_output -> out_anchor_index_of_parent_node
/// @return CompleteGraphBuilder
CompleteGraphBuilder &CompleteGraphBuilder::SetOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  for (auto &item : output_mapping) {
    output_mapping_[item.first] = item.second;
  }
  return *this;
}

/// @brief Build graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return ComputeGraphPtr
ComputeGraphPtr CompleteGraphBuilder::Build(graphStatus &error_code, std::string &error_msg) {
  owner_graph_ = ComGraphMakeShared<ComputeGraph>(name_);
  if (owner_graph_ == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return nullptr;
  }

  BuildNodes(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildDataLinks(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildCtrlLinks(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  AddDataNodes(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  if (retval_flag_) {
    AddRetValNodes(error_code, error_msg);
    if (error_code != af::GRAPH_SUCCESS) {
      return nullptr;
    }
    BuildGraphTargets(error_code, error_msg);
    if (error_code != af::GRAPH_SUCCESS) {
      return nullptr;
    }
  } else {
    AddNetOutputNode(error_code, error_msg);
    if (error_code != af::GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  PostProcess(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  return owner_graph_;
}

/// @brief Add data nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void CompleteGraphBuilder::AddDataNodes(graphStatus &error_code, std::string &error_msg) {
  for (auto &input : graph_inputs_) {
    const NodePtr data_node = AddDataNode(input.first, error_code, error_msg);
    if (data_node == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddDataNodes failed: add node Data:" + std::to_string(input.first) + +" failed.";
      return;
    }

    if (owner_graph_->AddInputNode(data_node) == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddDataNodes failed: add input node Data:" + std::to_string(input.first) + +" failed.";
      return;
    }

    // useless input
    const std::vector<std::string> input_names = input.second.first;
    const std::vector<uint32_t> anchor_indes = input.second.second;
    if (input_names.size() != anchor_indes.size()) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddDataNodes failed: num of input_names and indexes not equal.";
      return;
    }
    if (input_names.empty()) {
      continue;
    }

    const size_t input_num = input_names.size();
    for (size_t i = 0U; i < input_num; i++) {
      const std::string input_name = input_names[i];
      const int32_t ind = static_cast<int32_t>(anchor_indes[i]);
      const auto iter = node_names_.find(input_name);
      if (iter == node_names_.end()) {
        error_code = af::GRAPH_FAILED;
        error_msg = "AddDataNodes failed: node " + input_name + " does not exist in graph.";
        return;
      }

      const NodePtr in_node = node_names_[input_name];
      if (in_node == nullptr) {
        error_code = af::GRAPH_FAILED;
        error_msg = "AddDataNodes failed: node " + input_name + " is NULL.";
        return;
      }

      if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), in_node->GetInDataAnchor(ind)) != af::GRAPH_SUCCESS) {
        error_code = af::GRAPH_FAILED;
        error_msg = "AddDataNodes failed: add data-edge Data:" + std::to_string(input.first) + ":0->" + input_name +
                    ":" + std::to_string(ind) + " failed.";
        return;
      }
    }

    GELOGD("AddDataNodes : Add %u input succ.", input.first);
  }

  GELOGD("AddDataNodes succ.");
}

/// @brief Add data node
/// @param [in] index
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
NodePtr CompleteGraphBuilder::AddDataNode(const uint32_t index, graphStatus &error_code, std::string &error_msg) {
  const std::string data_name = "Data_" + std::to_string(index);
  OpDescBuilder op_desc_builder(data_name, "Data");
  const OpDescPtr op_desc = op_desc_builder.AddInput("x").AddOutput("y").Build();
  if (op_desc == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = "AddDataNode failed: create op_desc " + data_name + " failed.";
    return nullptr;
  }

  const auto index_iter = input_mapping_.find(index);
  if (index_iter != input_mapping_.end()) {
    if (!af::AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(index_iter->second))) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddDataNode failed: set attr ATTR_NAME_PARENT_NODE_INDEX for " + data_name + " failed.";
      return nullptr;
    }
  }
  if (parent_node_ != nullptr) {
    // op_desc should not be null
    const auto &parent_desc = parent_node_->GetOpDesc()->GetInputDesc(index_iter->second);
    if ((op_desc->UpdateInputDesc(0U, parent_desc) != af::GRAPH_SUCCESS) ||
        (op_desc->UpdateOutputDesc(0U, parent_desc) != af::GRAPH_SUCCESS)) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddDataNode failed: update tensor_desc for " + data_name + " failed.";
      return nullptr;
    }
  }

  const NodePtr data_node = owner_graph_->AddNode(op_desc);
  if (data_node == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = "AddDataNode failed: add node " + data_name + " failed.";
    return nullptr;
  }
  node_names_[data_name] = data_node;

  return data_node;
}

/// @brief Add RetVal nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void CompleteGraphBuilder::AddRetValNodes(graphStatus &error_code, std::string &error_msg) {
  const size_t output_num = graph_outputs_.size();
  for (size_t i = 0U; i < output_num; i++) {
    const int32_t index = static_cast<int32_t>(graph_outputs_[i].second);
    const auto out_iter = node_names_.find(graph_outputs_[i].first);
    if (out_iter == node_names_.end()) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode failed: node " + graph_outputs_[i].first + " does not exist in graph.";
      return;
    }
    const NodePtr node = out_iter->second;
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode failed: node is NULL.";
      return;
    }

    const std::string name = node->GetName() + "_RetVal_" + std::to_string(index);
    const OpDescPtr ret_val_desc = ComGraphMakeShared<OpDesc>(name, FRAMEWORKOP);
    if (ret_val_desc == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: op_desc is NULL.";
      return;
    }
    const af::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(index));
    if ((ret_val_desc->AddInputDesc(tensor) != af::GRAPH_SUCCESS) ||
        (ret_val_desc->AddOutputDesc(tensor) != af::GRAPH_SUCCESS)) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add input_desc / output_desc failed.";
      return;
    }

    if (!(af::AttrUtils::SetStr(ret_val_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal") &&
          af::AttrUtils::SetInt(ret_val_desc, RETVAL_ATTR_NAME_INDEX, static_cast<int64_t>(i)))) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: set FRAMEWORK_ORIGINAL_TYPE / RETVAL_ATTR_NAME_INDEX failed.";
      return;
    }
    const auto iter = output_mapping_.find(i);
    if (iter != output_mapping_.end()) {
      if (!af::AttrUtils::SetInt(ret_val_desc, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(iter->second))) {
        error_code = af::GRAPH_FAILED;
        error_msg = "AddRetValNode " + name + " failed: set attr PARENT_NODE_INDEX failed.";
        return;
      }
    }

    const NodePtr ret_val_node = owner_graph_->AddNode(ret_val_desc);
    if (ret_val_node == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add node failed.";
      return;
    }

    if (GraphUtils::AddEdge(node->GetOutDataAnchor(index), ret_val_node->GetInDataAnchor(0)) != af::GRAPH_SUCCESS) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add data-edge " + node->GetName() + ":" + std::to_string(index) +
                  "->" + ret_val_node->GetName() + ":0 failed.";
      return;
    }
  }

  GELOGD("AddRetValNodes succ.");
}

/// @brief Build target-nodes for graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void CompleteGraphBuilder::BuildGraphTargets(graphStatus &error_code, std::string &error_msg) {
  std::vector<NodePtr> target_nodes;
  for (const std::string &target_name : graph_targets_) {
    const auto target_iter = node_names_.find(target_name);
    if ((target_iter == node_names_.end()) || (target_iter->second == nullptr)) {
      error_code = af::GRAPH_FAILED;
      error_msg = "BuildGraphTargets failed: target_node " + target_name + " does not exist in graph.";
      return;
    }
    target_nodes.emplace_back(target_iter->second);
  }
  owner_graph_->SetGraphTargetNodesInfo(target_nodes);
}

/// @brief Add NetOutput node
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void CompleteGraphBuilder::AddNetOutputNode(graphStatus &error_code, std::string &error_msg) {
  if (graph_outputs_.empty() && graph_targets_.empty()) {
    return;
  }
  const std::string node_name = "Node_Output";
  const std::string log_msg = "AddNetOutputNode name:" + node_name + ", type:" + NETOUTPUT;
  const OpDescPtr net_output_desc = ComGraphMakeShared<OpDesc>(node_name, NETOUTPUT);
  if (net_output_desc == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = log_msg + " failed: op_desc is NULL.";
    return;
  }

  const size_t output_num = graph_outputs_.size();
  std::vector<OutDataAnchorPtr> peer_out_anchors(output_num);
  for (size_t i = 0U; i < output_num; i++) {
    const uint32_t index = graph_outputs_[i].second;
    const auto out_iter = node_names_.find(graph_outputs_[i].first);
    if (out_iter == node_names_.end()) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: node " + graph_outputs_[i].first + " does not exist in graph.";
      return;
    }
    const NodePtr node = out_iter->second;
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: node is NULL.";
      return;
    }

    af::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(index);
    int64_t update_index = static_cast<int64_t>(i);
    const auto iter = output_mapping_.find(i);
    if (iter != output_mapping_.end()) {
      update_index = static_cast<int64_t>(iter->second);
      (void)af::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, update_index);
    }

    if (net_output_desc->AddInputDesc(tensor) != af::GRAPH_SUCCESS) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add input_desc failed.";
      return;
    }
    peer_out_anchors[i] = node->GetOutDataAnchor(static_cast<int32_t>(index));
  }

  BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return;
  }

  GELOGD("%s succ.", log_msg.c_str());
}

/// @brief Build NetOutput nodes with data & ctrl edges
/// @param [in] net_output_desc
/// @param [in] peer_out_anchors
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void CompleteGraphBuilder::BuildNetOutputNodeWithLink(const OpDescPtr &net_output_desc,
                                                      const std::vector<OutDataAnchorPtr> &peer_out_anchors,
                                                      graphStatus &error_code, std::string &error_msg) {
  const std::string log_msg = "AddNetOutputNode name:" + std::string(NODE_NAME_NET_OUTPUT) + ", type:" + NETOUTPUT;
  const NodePtr net_output = owner_graph_->AddNode(net_output_desc);
  if (net_output == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = log_msg + " failed: add NetOutput node failed.";
    return;
  }
  owner_graph_->SetNetOutputNode(net_output);

  const size_t output_num = graph_outputs_.size();
  for (size_t i = 0U; i < output_num; i++) {
    if (GraphUtils::AddEdge(peer_out_anchors[i], net_output->GetInDataAnchor(static_cast<int32_t>(i))) !=
        af::GRAPH_SUCCESS) {
      error_code = af::GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add data-edge " + peer_out_anchors[i]->GetOwnerNode()->GetName() + ":" +
                  std::to_string(peer_out_anchors[i]->GetIdx()) + "->" + NODE_NAME_NET_OUTPUT + ":" +
                  std::to_string(i) + " failed.";
      return;
    }
  }
  for (const std::string &target_name : graph_targets_) {
    const auto target_iter = node_names_.find(target_name);
    if ((target_iter == node_names_.end()) || (target_iter->second == nullptr)) {
      error_code = af::GRAPH_FAILED;
      error_msg = "BuildGraphTargets failed: target_node " + target_name + " does not exist in graph.";
      return;
    }
    const auto &target_node = target_iter->second;
    if (GraphUtils::AddEdge(target_node->GetOutControlAnchor(), net_output->GetInControlAnchor()) !=
        af::GRAPH_SUCCESS) {
      error_code = af::GRAPH_FAILED;
      error_msg =
          "AddNetOutputNode failed: add ctrl-edge " + target_node->GetName() + "->" + NODE_NAME_NET_OUTPUT + " failed.";
      return;
    }
  }
}

/// @brief process after build
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void CompleteGraphBuilder::PostProcess(graphStatus &error_code, std::string &error_msg) {
  if (parent_node_ != nullptr) {
    owner_graph_->SetParentNode(parent_node_);
    const auto &parent_graph = parent_node_->GetOwnerComputeGraph();
    if (parent_graph == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "Parent graph is null, parent_node=" + parent_node_->GetName();
      return;
    }
    owner_graph_->SetParentGraph(parent_graph);
    // ATTR_NAME_SESSION_GRAPH_ID
    std::string graph_id;
    if ((!AttrUtils::GetStr(parent_graph, ATTR_NAME_SESSION_GRAPH_ID, graph_id)) ||
        (!AttrUtils::SetStr(owner_graph_, ATTR_NAME_SESSION_GRAPH_ID, graph_id))) {
      error_code = af::GRAPH_FAILED;
      error_msg = "Copy attr session_graph_id failed.";
      return;
    }
    if (parent_graph->HasAttr(ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED)) {
      bool is_dynamic_shape = false;
      if ((!AttrUtils::GetBool(parent_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape)) ||
          (!AttrUtils::SetBool(owner_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape))) {
        error_code = af::GRAPH_FAILED;
        error_msg = "Copy attr _dynamic_shape_partitioned failed.";
        return;
      }
    }
    owner_graph_->SetGraphUnknownFlag(parent_graph->GetGraphUnknownFlag());

    // refresh parent node/graph in subgraphs
    for (const NodePtr &node : owner_graph_->GetDirectNode()) {
      std::vector<ComputeGraphPtr> subgraphs;
      if (NodeUtils::GetDirectSubgraphs(node, subgraphs) != af::GRAPH_SUCCESS) {
        error_code = af::GRAPH_FAILED;
        error_msg = "Get subgraphs for node " + node->GetName() + " failed.";
        return;
      }
      for (const auto &subgraph : subgraphs) {
        subgraph->SetParentNode(node);
        subgraph->SetParentGraph(subgraph);
      }
    }
  }

  // refresh node name
  for (const NodePtr &node : owner_graph_->GetDirectNode()) {
    if ((node->GetOpDesc() == nullptr) || (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2)) {
      continue;
    }
    node->GetOpDesc()->SetName(owner_graph_->GetName() + "/" + node->GetName());
  }
}

/// @brief Add node to graph
/// @param [in] op_desc
/// @return PartialGraphBuilder
PartialGraphBuilder &PartialGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  (void)ComputeGraphBuilder::AddNode(op_desc);
  return *this;
}

/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return PartialGraphBuilder
PartialGraphBuilder &PartialGraphBuilder::AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                                      const std::string &dst_name, const uint32_t in_anchor_ind) {
  (void)ComputeGraphBuilder::AddDataLink(src_name, out_anchor_ind, dst_name, in_anchor_ind);
  return *this;
}

/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return PartialGraphBuilder
PartialGraphBuilder &PartialGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  (void)ComputeGraphBuilder::AddControlLink(src_name, dst_name);
  return *this;
}

/// @brief Set owner graph
/// @param [in] graph
/// @return PartialGraphBuilder
PartialGraphBuilder &PartialGraphBuilder::SetOwnerGraph(const ComputeGraphPtr &graph) {
  owner_graph_ = graph;
  return *this;
}

/// @brief Add exist node
/// @param [in] node
/// @return PartialGraphBuilder
PartialGraphBuilder &PartialGraphBuilder::AddExistNode(const NodePtr &exist_node) {
  exist_nodes_.emplace_back(exist_node);
  return *this;
}

/// @brief Build partial graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return ComputeGraphPtr
ComputeGraphPtr PartialGraphBuilder::Build(graphStatus &error_code, std::string &error_msg) {
  if (owner_graph_ == nullptr) {
    error_code = af::GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return nullptr;
  }

  BuildNodes(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildExistNodes(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildDataLinks(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildCtrlLinks(error_code, error_msg);
  if (error_code != af::GRAPH_SUCCESS) {
    return nullptr;
  }

  return owner_graph_;
}

/// @brief Build exist nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
void PartialGraphBuilder::BuildExistNodes(graphStatus &error_code, std::string &error_msg) {
  std::string node_name;
  for (auto &exist_node : exist_nodes_) {
    if (exist_node == nullptr) {
      error_code = af::GRAPH_FAILED;
      error_msg = "Build exist nodes failed: node is NULL.";
      return;
    }

    node_name = exist_node->GetName();
    if (exist_node->GetOwnerComputeGraph() != owner_graph_) {
      error_code = af::GRAPH_FAILED;
      error_msg = "Build exist nodes failed: node " + node_name + " not belongs to this graph.";
      return;
    }

    GELOGD("Add exist_node name:%s.", node_name.c_str());
    node_names_[node_name] = exist_node;
  }

  GELOGD("Build exist nodes succ.");
}
graphStatus GraphUtils::MoveNodeToGraph(const NodePtr &node, ComputeGraph &dst_graph) {
  GE_ASSERT_SUCCESS(IsolateNode(node, {}));
  GE_ASSERT_SUCCESS(RemoveNodesWithoutRelink(node->GetOwnerComputeGraph(), {node}));
  GE_ASSERT_NOTNULL(dst_graph.AddNode(node));
  GE_ASSERT_SUCCESS(node->SetOwnerComputeGraph(dst_graph.shared_from_this()));
  return af::GRAPH_SUCCESS;
}

graphStatus GraphUtils::RemoveJustNodes(const ComputeGraphPtr &compute_graph,
                                        const std::unordered_set<NodePtr> &nodes) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(compute_graph->impl_);

  size_t success_removed_nodes_size = 0U;
  for (auto iter = compute_graph->impl_->nodes_.begin(); iter != compute_graph->impl_->nodes_.end();) {
    if (nodes.count(*iter) > 0U) {
      GELOGD("Remove %s from graph %s.", (*iter)->GetNamePtr(), compute_graph->GetName().c_str());
      iter = compute_graph->impl_->nodes_.erase(iter);
      --(compute_graph->impl_->direct_nodes_size_);
      ++success_removed_nodes_size;
    } else {
      ++iter;
    }
  }
  const auto to_be_remove_nodes_size = nodes.size();
  if (success_removed_nodes_size != to_be_remove_nodes_size) {
    GELOGW("Successfully remove %zu nodes but there are %zu nodes to be deleted", success_removed_nodes_size,
           to_be_remove_nodes_size);
  }
  return af::GRAPH_SUCCESS;
}

namespace {
bool IsCurrentNodeHasMaxTopid(const NodePtr &node, const std::vector<Node *> out_nodes) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  int64_t cur_topid = node->GetOpDesc()->GetId();
  for (const auto &out_node : out_nodes) {
    GE_ASSERT_NOTNULL(out_node->GetOpDesc());
    if (out_node->GetOpDesc()->GetId() > cur_topid) {
      GELOGD("Current node %s does not have max topid.", node->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool HasSameStreamId(const NodePtr &node, const std::vector<Node *> out_nodes) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  int64_t cur_stream_id = node->GetOpDesc()->GetStreamId();
  std::string node_name = node->GetName();
  for (const auto &out_node : out_nodes) {
    GE_ASSERT_NOTNULL(out_node->GetOpDesc());
    auto out_node_stream_id = out_node->GetOpDesc()->GetStreamId();
    if (cur_stream_id == kInvalidStream) {
      cur_stream_id = out_node_stream_id;
      node_name = out_node->GetName();
      continue;
    }
    if (out_node_stream_id != cur_stream_id && out_node_stream_id != kInvalidStream) {
      GELOGD("Node %s stream id[%lld] is not same with node %s stream id[%lld].", node_name.c_str(), cur_stream_id,
             out_node->GetName().c_str(), out_node_stream_id);
      return false;
    }
  }
  return true;
}

bool HasRefAttr(const std::vector<Node *> out_nodes) {
  for (const auto &out_node : out_nodes) {
    GE_ASSERT_NOTNULL(out_node->GetOpDesc());
    bool is_ref = false;
    (void)af::AttrUtils::GetBool(out_node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);
    if (is_ref) {
      GELOGD("Node %s has ref attr.", out_node->GetName().c_str());
      return true;
    }
  }
  return false;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::GetSupportInplaceOutput(
    const NodePtr &node, std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes) {
  // 找到带output_inplace_ability属性的节点
  GE_ASSERT_NOTNULL(node);
  auto node_op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_op_desc);
  // 获取output_inplace_ability属性,Ge拿到的就是真实的index，不是ir index
  std::vector<std::vector<int64_t>> output_inplace_index_list;
  if (!af::AttrUtils::GetListListInt(node_op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, output_inplace_index_list)) {
    return af::GRAPH_SUCCESS;
  }
  // 类似这种键值对：{{0,0},{0,1},{0,3},{1,4}}
  constexpr size_t kInplaceAbilitySize = 2U;
  for (auto &inplace_index : output_inplace_index_list) {
    if (inplace_index.size() != kInplaceAbilitySize) {
      GELOGW("The size %zu of inplace index is invalid, must be equal to 2.", inplace_index.size());
      return af::GRAPH_FAILED;
    }
    GE_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(inplace_index[0]));
    GE_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(inplace_index[1]));
    size_t output_index = inplace_index[0];
    size_t input_index = inplace_index[1];
    // 判断输入节点对应的所有输出之中,当前节点的topid是否最大，streamid是否相同
    auto in_node = NodeUtils::GetInDataNodeByIndex(*node, static_cast<int32_t>(input_index));
    GE_ASSERT_NOTNULL(in_node);
    GE_ASSERT_NOTNULL(node->GetOutDataAnchor(static_cast<int32_t>(output_index)));
    auto out_nodes = in_node->GetOutDataNodesPtr();
    GELOGD("Check whether node %s's %zu output can be inplaced, input node is %s input_index[%zu].",
           node->GetName().c_str(), output_index, in_node->GetName().c_str(), input_index);
    if (IsCurrentNodeHasMaxTopid(node, out_nodes) && HasSameStreamId(in_node, out_nodes) && !HasRefAttr(out_nodes)) {
      // 一个输出对应多个输入
      out_index_to_refable_in_indexes[output_index].push_back(input_index);
    }
  }

  for (auto &item : out_index_to_refable_in_indexes) {
    for (auto &index : item.second) {
      GELOGD("Node %s's output[%zu] can inplace input[%zu].", node->GetName().c_str(), item.first, index);
    }
  }

  return af::GRAPH_SUCCESS;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr
GraphUtils::BuildGraphFromNodes(const std::unordered_set<NodePtr> &nodes, const std::string &name) {
  if (nodes.empty()) {
    GELOGW("nodes are empty, no need to build subgraph");
    return nullptr;
  }

  GraphInfo graph_info;
  BuildGraphInfoFromNodes(nodes, graph_info);
  return BuildGraph(graph_info, name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr GraphUtils::BuildGraphInternal(
    const GraphUtils::GraphInfo &graph_info, const string &name, const NodePtr &parent_node) {
  // 构造 graph_builder
  CompleteGraphBuilder graph_builder(name, false);

  // 如果是子图，设置父节点
  if (parent_node != nullptr) {
    (void)graph_builder.SetParentNode(parent_node);
  }

  // 添加节点
  for (const auto &node : graph_info.nodes_) {
    (void)graph_builder.AddNode(GraphUtils::CopyOpDesc(node->GetOpDesc()));
  }

  // 设置输入
  uint32_t index = 0U;
  std::map<uint32_t, uint32_t> input_mapping;
  for (const auto &item : graph_info.data_inputs_) {
    for (const auto &in_data_anchor : item.second.second) {
      (void)graph_builder.SetInput(index, {in_data_anchor->GetOwnerNodeBarePtr()->GetName()},
                                   {static_cast<uint32_t>(in_data_anchor->GetIdx())});
      if (parent_node != nullptr) {  // 仅子图需要设置输入映射
        input_mapping[index] = index;
      }
      index++;
    }
  }

  // 设置输入映射（仅子图）
  if (parent_node != nullptr) {
    (void)graph_builder.SetInputMapping(input_mapping);
  }

  // 添加输出
  index = 0U;
  std::map<uint32_t, uint32_t> output_mapping;
  for (const auto &item : graph_info.data_outputs_) {
    (void)graph_builder.AddOutput(item.second.first->GetOwnerNodeBarePtr()->GetName(),
                                  static_cast<uint32_t>(item.second.first->GetIdx()));
    if (parent_node != nullptr) {  // 仅子图需要设置输出映射
      output_mapping[index] = index;
    }
    index++;
  }

  // 设置输出映射（仅子图）
  if (parent_node != nullptr) {
    (void)graph_builder.SetOutputMapping(output_mapping);
  }

  // 添加目标节点
  for (const auto &item : graph_info.ctrl_outputs_) {
    (void)graph_builder.AddTarget(item.first->GetOwnerNodeBarePtr()->GetName());
  }

  // 添加数据边
  for (const auto &data_edge : graph_info.inner_data_edges_) {
    (void)graph_builder.AddDataLink(
        data_edge.first->GetOwnerNodeBarePtr()->GetName(), static_cast<uint32_t>(data_edge.first->GetIdx()),
        data_edge.second->GetOwnerNodeBarePtr()->GetName(), static_cast<uint32_t>(data_edge.second->GetIdx()));
  }

  // 添加控制边
  for (const auto &ctrl_edge : graph_info.inner_ctrl_edges_) {
    (void)graph_builder.AddControlLink(ctrl_edge.first->GetOwnerNodeBarePtr()->GetName(),
                                       ctrl_edge.second->GetOwnerNodeBarePtr()->GetName());
  }

  // 构建图
  graphStatus error_code = af::GRAPH_SUCCESS;
  std::string error_msg;
  auto graph = graph_builder.Build(error_code, error_msg);
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Build graph %s failed:%s.", name.c_str(), error_msg.c_str());
    GELOGE(error_code, "[Build][Graph] %s failed:%s.", name.c_str(), error_msg.c_str());
  }
  return graph;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::GenDumpOnnxFileName(
    const ComputeGraphPtr &compute_graph, const std::string &suffix, std::string &real_path_name) {
#ifdef FMK_SUPPORT_DUMP
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump ge onnx file: %" PRId64, file_index);
  if (CheckDumpGraphNum(file_index) != af::GRAPH_SUCCESS) {
    return FAILED;
  }

  std::stringstream stream_file_name;
  GetDumpGraphPrefix(stream_file_name);
  if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
    const int32_t ret = CreateDir(stream_file_name.str());
    if (ret != 0) {
      GELOGW("[DumpGraph][CreateDir] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
      stream_file_name.str("");
      stream_file_name << "./";
    }
  }

  std::string single_op = "";
  if (IsSingleOpScene(compute_graph)) {
    single_op = "_aclop";
  }
  std::stringstream ss;
  ss << "ge_onnx_" << std::setw(kDumpGraphIndexWidth) << std::setfill('0') << file_index << single_op;
  ss << "_graph_" << compute_graph->GetGraphID() << "_" << GetSanitizedName(suffix) << ".pbtxt";
  std::string dump_file_name = ss.str();
  if ((dump_file_name.length()) >= kNameMax) {
    const size_t original_length = dump_file_name.length();
    dump_file_name = dump_file_name.substr(0U, kNameMax - 7U) + ".pbtxt";
    GELOGW("[Check][Param] File name is too long!, length:%zu, max:%zu, file:%s", original_length, kNameMax,
           dump_file_name.c_str());
  }
  std::string proto_file = stream_file_name.str() + dump_file_name;

  char_t real_path[MMPA_MAX_PATH] = {};
  /// Returning nullptr means 3 case as follows:
  /// a.path is MMPA_MAX_PATH chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  auto const ret = mmRealPath(proto_file.c_str(), &(real_path[0]), MMPA_MAX_PATH);
  if (ret != EN_OK) {
    GELOGD("[Get][RealPath] file does not exist, it will be created. ret:%d", ret);
  }
  real_path_name = real_path;
  return SUCCESS;
#else
  (void)compute_graph;
  (void)suffix;
  (void)real_path_name;
  GELOGW("[Gen][OnnxFileName] Need to define FMK_SUPPORT_DUMP for dump graph.");
  return FAILED;
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::GenDumpTxtFileName(const ComputeGraphPtr &compute_graph, const std::string &suffix,
                               const std::string &user_graph_name, std::string &real_path_name) {
#ifdef FMK_SUPPORT_DUMP
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump om txt: %" PRId64, file_index);
  if (CheckDumpGraphNum(file_index) != af::GRAPH_SUCCESS) {
    return FAILED;
  }
  std::stringstream stream_file_name;
  std::string single_op = "";
  if (IsSingleOpScene(compute_graph)) {
    single_op = "aclop_";
  }
  stream_file_name << single_op << "graph_" << compute_graph->GetGraphID() << "_"
                   << suffix;  // add graphId, like graph_x_xxxx
  auto const ret = GetDumpRealPath(file_index, stream_file_name.str(), user_graph_name, real_path_name);
  if (ret != af::GRAPH_SUCCESS) {
    GELOGW("[Get][RealPath] Get real path failed, ret:%d, file_index:%" PRId64 ", name:%s", ret, file_index,
           stream_file_name.str().c_str());
    return FAILED;
  }
  return af::GRAPH_SUCCESS;
#else
  (void)compute_graph;
  (void)suffix;
  (void)user_graph_name;
  (void)real_path_name;
  GELOGW("[Gen][TxtFileName] Need to define FMK_SUPPORT_DUMP for dump graph.");
  return FAILED;
#endif
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::GenDumpReadableTxtFileName(const ComputeGraphPtr &compute_graph, const std::string &suffix,
                                       const std::string &user_graph_name, std::string &real_path_name) {
#ifdef FMK_SUPPORT_DUMP
  // file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump readable graph txt: %" PRId64, file_index);

  std::stringstream stream_file_name;
  std::string single_op = "";
  if (IsSingleOpScene(compute_graph)) {
    single_op = "aclop_";
  }
  stream_file_name << single_op << "graph_" << compute_graph->GetGraphID() << "_"
                   << suffix;  // add graphId, like graph_x_xxxx
  auto const ret = GetDumpRealPath(file_index, stream_file_name.str(), user_graph_name, real_path_name, "ge_readable_");
  if (ret != af::GRAPH_SUCCESS) {
    GELOGW("[Get][RealPath] Get real path failed, ret:%d, file_index:%" PRId64 ", name:%s", ret, file_index,
           stream_file_name.str().c_str());
    return FAILED;
  }

  return af::GRAPH_SUCCESS;
#else
  (void)compute_graph;
  (void)suffix;
  (void)user_graph_name;
  (void)real_path_name;
  GELOGW("[Gen][TxtFileName] Need to define FMK_SUPPORT_DUMP for dump graph.");
  return FAILED;
#endif
}
}  // namespace af
