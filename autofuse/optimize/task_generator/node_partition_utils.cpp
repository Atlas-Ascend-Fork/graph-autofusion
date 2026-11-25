/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_partition_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_ops.h"
#include "schedule_utils.h"
#include <algorithm>
#include <map>

namespace optimize {
namespace {
std::string GetNewNodeName(const af::AscNodePtr &src_node, const af::AscNodePtr &dst_node, const std::string &type,
                           int32_t idx) {
  return src_node->GetName() + "_to_" + dst_node->GetName() + "_" + type + "_" + to_string(idx);
}

Status PartitionLoadNode(af::AscNodePtr &src_load_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph) {
  // 取数据输入而非 GetInNodes: 后者混入控制边, 且空容器时 at() 抛异常直接终止进程
  const auto load_data_nodes = src_load_node->GetInDataNodes();
  GE_ASSERT_TRUE(!load_data_nodes.empty(), "Load node [%s] has no data input.", src_load_node->GetNamePtr());
  auto load_input_asc_node = std::dynamic_pointer_cast<af::AscNode>(load_data_nodes.at(0UL));
  GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Data>(load_input_asc_node) ||
                 af::ops::IsOps<af::ascir_op::Workspace>(load_input_asc_node));
  af::ascir_op::Load load(("copy_from_" + src_load_node->GetName()).c_str());

  af::AscNodePtr new_load_input_node;
  if (af::ops::IsOps<af::ascir_op::Data>(load_input_asc_node)) {
    // Data 的 GM 身份是 ir_attr 里的输入参数 index，随属性克隆保留，改名不影响寻址。
    af::ascir_op::Data data(("copy_from_" + load_input_asc_node->GetName()).c_str());
    new_load_input_node = impl_graph.AddNode(data);
  } else {
    // Workspace 没有 index 这样的身份属性，它的 GM 身份就是节点名：BufQueAllocator 按
    // 名字分配 tensor_id，tiling 再按 tensor_id 给每个 id 一段互不相交的 workspace 偏移。
    // 副本改名会多分出一块谁也没写过的 GM，所以副本必须沿用源 workspace 的名字。
    // 同一子图内由此产生的同名无输入副本，由 MergeDuplicatedCopyNodesInSubGraph 合并。
    af::ascir_op::Workspace workspace(load_input_asc_node->GetName().c_str());
    new_load_input_node = impl_graph.AddNode(workspace);
  }
  auto load_node = impl_graph.AddNode(load);
  DoCopyAscNodeTensorAttr(load_input_asc_node, new_load_input_node);
  DoCopyAscNodeTensorAttr(src_load_node, load_node);
  // dst_node 的多个输入可能来自同一个 src_load_node（如 Mul 的两个输入都是同一个 Load），
  // 需要遍历所有 peer 边逐一断开并替换，不能找到第一条就 return。
  // new_load_input -> load_node 的边只需建一次，后续匹配的 peer 边共用这条链路。
  bool found = false;
  for (const auto &out_anchor : src_load_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_load_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_load_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        if (!found) {
          found = true;
          GE_CHK_STATUS_RET(
              af::GraphUtils::AddEdge(new_load_input_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
        }
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_load_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status PartitionScalarNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph) {
  af::AscNodePtr scalar_node;
  if (af::ops::IsOps<af::ascir_op::ScalarData>(src_node)) {
    af::ascir_op::ScalarData scalar_data(("copy_from_" + src_node->GetName()).c_str());
    scalar_node = impl_graph.AddNode(scalar_data);
  } else {
    af::ascir_op::Scalar scalar(("copy_from_" + src_node->GetName()).c_str());
    scalar_node = impl_graph.AddNode(scalar);
  }
  DoCopyAscNodeTensorAttr(src_node, scalar_node);
  // dst_node 的多个输入可能来自同一个 src_node，需要遍历所有 peer 边逐一断开并替换，不能找到第一条就 return。
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(scalar_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status PartitionDataNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph) {
  // Data 是 GM 输入占位符（kAPITypeBuffer），值直接来自 kernel 输入参数，不能走通用路径用 Store 物化
  // （Data 直连 Store 会生成 DataCopyPadExtend(GlobalTensor, ...) 无匹配重载的非法调用）。
  // 复制 Data 节点让 citation 侧子图重读同一 GM 输入，ir_attr 携带的输入参数 index 随属性克隆保留。
  af::ascir_op::Data data(("copy_from_" + src_node->GetName()).c_str());
  auto data_node = impl_graph.AddNode(data);
  GE_CHK_STATUS_RET(DoCopyAscNodeTensorAttr(src_node, data_node));
  // dst_node 的多个输入可能来自同一个 src_node，需要遍历所有 peer 边逐一断开并替换，不能找到第一条就 return。
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(data_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status PartitionGenericNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph) {
  for (const auto &out_anchor : src_node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor, "Out data anchor is null, node:%s.", src_node->GetNamePtr());
    // 仅对实际连到 dst 的输出建链: 不连 dst 的输出若也建 ws 链, 会生成无消费者的孤儿
    // ws_post->load 分量, 污染 PartitionByConnectivity 的分组结果
    bool has_dst_peer = false;
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        has_dst_peer = true;
        break;
      }
    }
    if (!has_dst_peer) {
      continue;
    }
    af::ascir_op::Workspace workspace_pre(
        GetNewNodeName(src_node, dst_node, "Workspace", out_anchor->GetIdx()).c_str());
    af::ascir_op::Workspace workspace_post(
        GetNewNodeName(src_node, dst_node, "Workspace", out_anchor->GetIdx()).c_str());
    af::ascir_op::Load load(GetNewNodeName(src_node, dst_node, "Load", out_anchor->GetIdx()).c_str());
    af::ascir_op::Store store(GetNewNodeName(src_node, dst_node, "Store", out_anchor->GetIdx()).c_str());
    auto workspace_pre_node = impl_graph.AddNode(workspace_pre);
    auto workspace_post_node = impl_graph.AddNode(workspace_post);
    auto load_node = impl_graph.AddNode(load);
    auto store_node = impl_graph.AddNode(store);
    GE_CHK_STATUS_RET(DoCopyAscNodeOutputTensorAttr(src_node, load_node, out_anchor->GetIdx()));
    GE_CHK_STATUS_RET(DoCopyAscNodeOutputTensorAttr(src_node, store_node, out_anchor->GetIdx()));
    GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(store_node, workspace_pre_node));
    GE_CHK_STATUS_RET(DoCopyWorkspaceTensorAttr(load_node, workspace_post_node));
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNodeBarePtr(), "Peer in node:%s is null", src_node->GetNamePtr());
      if (peer_in_anchor->GetOwnerNodeBarePtr() == dst_node.get()) {
        // remove src->dst
        GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(src_node->GetOutAnchor(out_anchor->GetIdx()),
                                                     dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
        GE_CHK_STATUS_RET(
            af::GraphUtils::AddEdge(load_node->GetOutAnchor(0UL), dst_node->GetInAnchor(peer_in_anchor->GetIdx())));
      }
    }
    // add src->store->workspace_pre_node
    GE_CHK_STATUS_RET(
        af::GraphUtils::AddEdge(src_node->GetOutAnchor(out_anchor->GetIdx()), store_node->GetInAnchor(0UL)));
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(store_node->GetOutAnchor(0UL), workspace_pre_node->GetInAnchor(0UL)));
    // add workspace_post_node->load->dst
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(workspace_post_node->GetOutAnchor(0UL), load_node->GetInAnchor(0UL)));
  }
  return ge::GRAPH_SUCCESS;
}

// 可以在同一子图内合并的 GM 源副本：
//   - `copy_from_*`: 多引用切分复制出的 Data / ScalarData / Scalar 副本；
//   - 无输入的 Workspace: workspace 的 GM 身份就是节点名(BufQueAllocator 按名字分配
//     tensor_id)，所以 Workspace 副本沿用源名字、认不出 `copy_from_` 前缀。同名即同一块
//     GM，同一子图里重复读没有意义。带输入的那个是写入者，不在此列。
bool IsMergeableGmSourceCopy(const af::AscNodePtr &node) {
  if (node->GetName().rfind("copy_from_", 0U) == 0U) {
    return true;
  }
  return af::ops::IsOps<af::ascir_op::Workspace>(node) && (node->GetInDataNodesSize() == 0UL);
}

// 失去消费者之后可以摘掉的副本。`copy_from_*` 判据同上; 无输入的 Workspace 另加一条
// 「子图里还有同名节点」: 副本是被合并掉的那一份, 基准那一份仍在, 所以同名节点必然不止
// 一个。缺了这一条就会连带摘掉本来就孤立的 workspace 节点——那不是本次切分造出来的，
// 由图自己的悬挂节点清理负责。
bool IsRemovableDeadGmSourceCopy(const ascir::ImplGraph &sub_graph, const af::AscNodePtr &node) {
  if (!node->GetOutDataNodes().empty()) {
    return false;
  }
  if (node->GetName().rfind("copy_from_", 0U) == 0U) {
    return true;
  }
  if (!IsMergeableGmSourceCopy(node)) {
    return false;
  }
  size_t same_name = 0UL;
  for (const auto &other : sub_graph.GetAllNodes()) {
    if (other->GetName() == node->GetName()) {
      ++same_name;
    }
  }
  return same_name > 1UL;
}
}  // namespace

Status DoCopyAscNodeTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &dst_node) {
  auto op_desc = dst_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  for (size_t i = 0U; i < src_node->outputs().size(); i++) {
    GE_CHECK_NOTNULL(op_desc->MutableOutputDesc(i));
    auto tensor_attr_group = op_desc->MutableOutputDesc(i)->GetOrCreateAttrsGroup<af::AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = src_node->outputs[i].attr;
  }
  return af::SUCCESS;
}

Status DoCopyAscNodeOutputTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &dst_node,
                                     const uint32_t src_output_index) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(dst_node);
  GE_ASSERT_TRUE(src_output_index < src_node->outputs().size());
  GE_ASSERT_TRUE(!dst_node->outputs().empty());
  auto op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    dst_asc_node_attr->sched = src_asc_node_attr->sched;
    if (src_asc_node_attr->ir_attr) {
      dst_asc_node_attr->ir_attr = src_asc_node_attr->ir_attr->Clone();
    }
  }
  const auto output_desc = op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(output_desc);
  auto tensor_attr_group = output_desc->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_ASSERT_NOTNULL(tensor_attr_group);
  *tensor_attr_group = src_node->outputs[src_output_index].attr;
  return af::SUCCESS;
}

Status DoCopyWorkspaceTensorAttr(const af::AscNodePtr &src_node, af::AscNodePtr &workspace_node) {
  GE_ASSERT_NOTNULL(src_node);
  GE_ASSERT_NOTNULL(workspace_node);
  GE_ASSERT_TRUE(!src_node->outputs().empty());
  GE_ASSERT_TRUE(!workspace_node->outputs().empty());
  workspace_node->outputs[0].attr.dtype = src_node->outputs[0].attr.dtype;
  return af::SUCCESS;
}

bool IsFirstAxisBroadcast(const af::AscNodePtr &pre_node, const af::AscNodePtr &broadcast_node) {
  GE_ASSERT_NOTNULL(pre_node);
  GE_ASSERT_NOTNULL(broadcast_node);
  GE_ASSERT_TRUE(!pre_node->outputs().empty());
  GE_ASSERT_TRUE(!broadcast_node->outputs().empty());
  const auto &pre_repeats = pre_node->outputs[0].attr.repeats;
  const auto &broadcast_repeats = broadcast_node->outputs[0].attr.repeats;
  if (pre_repeats.empty() || pre_repeats.size() != broadcast_repeats.size()) {
    return false;
  }
  return af::SymbolicUtils::StaticCheckEq(pre_repeats[0], broadcast_repeats[0]) != af::TriBool::kTrue;
}

bool IsOnlyFirstAxisBroadcast(const af::AscNodePtr &pre_node, const af::AscNodePtr &broadcast_node) {
  if (!IsFirstAxisBroadcast(pre_node, broadcast_node)) {
    return false;
  }
  const auto &pre_repeats = pre_node->outputs[0].attr.repeats;
  const auto &broadcast_repeats = broadcast_node->outputs[0].attr.repeats;
  for (size_t i = 1UL; i < pre_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(pre_repeats[i], broadcast_repeats[i]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

Status PartitionByNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph,
                       bool &partition_flag, std::vector<af::AscNodePtr> &node_order) {
  partition_flag = true;
  node_order.emplace_back(src_node);
  if (ScheduleUtils::IsLoad(src_node)) {
    return PartitionLoadNode(src_node, dst_node, impl_graph);
  }
  if (ScheduleUtils::IsScalarLikeNode(src_node)) {
    return PartitionScalarNode(src_node, dst_node, impl_graph);
  };
  if (af::ops::IsOps<af::ascir_op::Data>(src_node)) {
    return PartitionDataNode(src_node, dst_node, impl_graph);
  }
  return PartitionGenericNode(src_node, dst_node, impl_graph);
}

// 手工注释：合并子图内同名的比如copy_from_scalar节点，避免codegen阶段检测失败
// 合并同一子图内的同名 copy_from_* 副本: 同一子图即同一连通分量/同一 kernel, 同名同类型
// 副本必然同源, 保留 id 最小者为基准, 其余副本的消费者改接到基准后摘除整节点; 副本摘除后
// 其输入副本(Data/Workspace)可能失去唯一消费者, 循环摘除直至无死副本。
// 由 PartitionByConnectivity 在子图划分定型后、CheckDuplicateWorkspaceNodes 之前逐子图
// 调用: 沿用源名字的 Workspace 副本与源节点同名, 落在同一子图时先合并掉才不会被重名检查
// 当成切分出错。合并只摘除冗余副本, 不改变子图归属, 因此不影响连通性。
Status MergeDuplicatedCopyNodesInSubGraph(ascir::ImplGraph &sub_graph) {
  std::map<std::pair<std::string, std::string>, std::vector<af::AscNodePtr>> copies_by_key;
  for (const auto &node : sub_graph.GetAllNodes()) {
    if (!IsMergeableGmSourceCopy(node)) {
      continue;
    }
    copies_by_key[std::make_pair(node->GetName(), node->GetType())].push_back(node);
  }
  for (auto &item : copies_by_key) {
    std::vector<af::AscNodePtr> &copies = item.second;
    if (copies.size() <= 1UL) {
      continue;
    }
    std::sort(copies.begin(), copies.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
      return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
    });
    const af::AscNodePtr &canonical = copies.front();
    for (size_t i = 1UL; i < copies.size(); ++i) {
      GELOGD("Merge duplicated copy node [%s](id=%lld) into [%s](id=%lld), sub graph [%s].", copies[i]->GetNamePtr(),
             copies[i]->GetOpDescBarePtr()->GetId(), canonical->GetNamePtr(), canonical->GetOpDescBarePtr()->GetId(),
             sub_graph.GetName().c_str());
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveNode(sub_graph, copies[i], canonical->GetOutDataAnchor(0UL)),
                        "Merge duplicated copy node [%s] failed.", copies[i]->GetName().c_str());
    }
    GELOGI("Merged %zu duplicated copy node(s) into [%s](id=%lld), sub graph [%s].", copies.size() - 1UL,
           canonical->GetNamePtr(), canonical->GetOpDescBarePtr()->GetId(), sub_graph.GetName().c_str());
  }
  bool has_removed = false;
  do {
    has_removed = false;
    std::vector<af::AscNodePtr> dead_copies;
    for (const auto &node : sub_graph.GetAllNodes()) {
      if (IsRemovableDeadGmSourceCopy(sub_graph, node)) {
        dead_copies.push_back(node);
      }
    }
    for (const auto &node : dead_copies) {
      GELOGD("Remove dead copy node [%s](id=%lld), sub graph [%s].", node->GetNamePtr(),
             node->GetOpDescBarePtr()->GetId(), sub_graph.GetName().c_str());
      GE_CHK_STATUS_RET(ScheduleUtils::RemoveNode(sub_graph, node, node->GetOutDataAnchor(0UL)),
                        "Remove dead copy node [%s] failed.", node->GetName().c_str());
      has_removed = true;
    }
  } while (has_removed);
  return af::SUCCESS;
}

}  // namespace optimize
