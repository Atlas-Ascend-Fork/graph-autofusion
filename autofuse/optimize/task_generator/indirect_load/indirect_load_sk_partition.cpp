/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

af::Status CopyBoundaryTensorAttr(const af::AscNodePtr &src_node, size_t src_output_idx,
                                  const af::AscNodePtr &dst_node) {
  GE_ASSERT_TRUE(src_output_idx < src_node->outputs().size(),
                 "IndirectLoad SK boundary output index %zu is out of range for node[%s].", src_output_idx,
                 src_node->GetNamePtr());
  GE_ASSERT_TRUE(!dst_node->outputs().empty(), "IndirectLoad SK boundary node[%s] has no output.",
                 dst_node->GetNamePtr());
  auto dst_op_desc = dst_node->GetOpDesc();
  GE_ASSERT_NOTNULL(dst_op_desc);
  auto dst_node_attr = dst_op_desc->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  auto src_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<af::AscNodeAttr>();
  GE_ASSERT_NOTNULL(dst_node_attr);
  if (src_node_attr != nullptr) {
    dst_node_attr->sched = src_node_attr->sched;
    if (src_node_attr->ir_attr != nullptr) {
      dst_node_attr->ir_attr = src_node_attr->ir_attr->Clone();
    }
  }
  auto output_desc = dst_op_desc->MutableOutputDesc(0UL);
  GE_ASSERT_NOTNULL(output_desc);
  auto output_attr = output_desc->GetOrCreateAttrsGroup<af::AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  *output_attr = src_node->outputs()[src_output_idx]->attr;
  return af::SUCCESS;
}

af::Status CopyWorkspaceTensorAttr(const af::AscNodePtr &boundary_node, const af::AscNodePtr &workspace_node) {
  GE_ASSERT_TRUE(!boundary_node->outputs().empty(), "IndirectLoad SK boundary node[%s] has no output.",
                 boundary_node->GetNamePtr());
  GE_ASSERT_TRUE(!workspace_node->outputs().empty(), "IndirectLoad SK workspace node[%s] has no output.",
                 workspace_node->GetNamePtr());
  workspace_node->outputs()[0]->attr = boundary_node->outputs()[0]->attr;
  return af::SUCCESS;
}

af::Status InsertWorkspaceBoundary(af::AscGraph &graph, const af::AscNodePtr &src_node, size_t src_output_idx,
                                   const af::AscNodePtr &dst_node, size_t dst_input_idx,
                                   const std::string &boundary_name, bool align_store, bool align_load) {
  const auto src_anchor = src_node->GetOutDataAnchor(src_output_idx);
  const auto dst_anchor = dst_node->GetInDataAnchor(dst_input_idx);
  GE_ASSERT_NOTNULL(src_anchor, "IndirectLoad SK source anchor is null for boundary[%s].", boundary_name.c_str());
  GE_ASSERT_NOTNULL(dst_anchor, "IndirectLoad SK destination anchor is null for boundary[%s].", boundary_name.c_str());
  GE_ASSERT_TRUE(dst_anchor->GetPeerOutAnchor() == src_anchor,
                 "IndirectLoad SK boundary[%s] does not match edge %s:%zu -> %s:%zu.", boundary_name.c_str(),
                 src_node->GetNamePtr(), src_output_idx, dst_node->GetNamePtr(), dst_input_idx);

  const std::string workspace_name = boundary_name + "_workspace";
  af::ascir_op::Workspace workspace_pre(workspace_name.c_str());
  af::ascir_op::Workspace workspace_post(workspace_name.c_str());
  af::ascir_op::Load load((boundary_name + "_load").c_str());
  af::ascir_op::Store store((boundary_name + "_store").c_str());
  auto workspace_pre_node = graph.AddNode(workspace_pre);
  auto workspace_post_node = graph.AddNode(workspace_post);
  auto load_node = graph.AddNode(load);
  auto store_node = graph.AddNode(store);
  GE_ASSERT_NOTNULL(workspace_pre_node);
  GE_ASSERT_NOTNULL(workspace_post_node);
  GE_ASSERT_NOTNULL(load_node);
  GE_ASSERT_NOTNULL(store_node);

  GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(src_node, src_output_idx, load_node));
  GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(src_node, src_output_idx, store_node));
  GE_ASSERT_SUCCESS(CopyWorkspaceTensorAttr(store_node, workspace_pre_node));
  GE_ASSERT_SUCCESS(CopyWorkspaceTensorAttr(load_node, workspace_post_node));
  if (align_store) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        store_node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }
  if (align_load) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        load_node, ascgen_utils::indirect_load::TemplateRole::kStridedUbPath));
  }

  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(src_anchor, dst_anchor));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(src_anchor, store_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::AddEdge(store_node->GetOutDataAnchor(0UL), workspace_pre_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(
      af::GraphUtils::AddEdge(workspace_post_node->GetOutDataAnchor(0UL), load_node->GetInDataAnchor(0UL)));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0UL), dst_anchor));
  return af::SUCCESS;
}

af::Status PartitionSkGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool align_input_path,
                            bool align_index_path, bool skip_input_boundaries) {
  // [SK段SIMT化] 窗口超限形态：input/index 不插 workspace 边界（IL 段以 SIMT 语义
  // 从 GM 直读原输入），仅保留 output 边界连接消费子图（Reduce/Norm 段独立调度，
  // 承载 R 轴分 tile 归约）。
  if (!skip_input_boundaries) {
    for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
      const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
      GE_ASSERT_NOTNULL(input_anchor);
      const auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
      GE_ASSERT_NOTNULL(peer_out_anchor);
      auto producer = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
      GE_ASSERT_NOTNULL(producer);
      const std::string role = input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? "input" : "index";
      const bool align_path =
          input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? align_input_path : align_index_path;
      GE_ASSERT_SUCCESS(
          InsertWorkspaceBoundary(graph, producer, static_cast<size_t>(peer_out_anchor->GetIdx()), indirect_load,
                                  input_idx, indirect_load->GetName() + "_sk_" + role, align_path,
                                  input_idx == ascgen_utils::indirect_load::kIndexTensorIndex && align_path));
    }
  }  // skip_input_boundaries
  const auto output_anchor = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(output_anchor);
  const auto peer_input_anchors = output_anchor->GetPeerInDataAnchors();
  // [SK段SIMT化] IL 输出可能多消费者（LayerNorm 的 x 与 x-mean 菱形：IL 直连
  // Reduce 与 Elementwise 分支，SIMT 流程的前置改写会折叠过渡 Cast）。逐消费者
  // 边独立边界化：IL 输出写 N 份 workspace，各消费者从对应 workspace 读取——
  // 语义等价（数据相同），不要求单消费者。
  const auto &il_output_anchors = peer_input_anchors;
  GE_ASSERT_TRUE(!il_output_anchors.empty(), "IndirectLoad SK requires at least one output consumer, node[%s].",
                 indirect_load->GetNamePtr());
  // [SK段SIMT化] IL 输出可能多消费者（LayerNorm 的 x 与 x-mean 菱形）。单份
  // workspace：首个消费者走完整边界（IL→store→workspace，workspace→load→consumer），
  // 其余消费者的输入边改接到该 load 输出——共享同一份数据，不产生多副本 load。
  // [SK段SIMT化] 多消费者时单份产出、多份读取：SIMT body 的 OutputTargets 仅支持
  // 单个 GM 输出（kGmOutputCount=1），IL 段只建一个 store/pre（写一份 workspace）；
  // 每个消费段各建独立的 post/load（同名 workspace 读同一 GM buffer，post 之间无
  // 边），连通性分量互不连通。单消费者时退化为标准 InsertWorkspaceBoundary。
  const std::string boundary_name = indirect_load->GetName() + "_sk_output";
  const auto first_peer = *il_output_anchors.begin();
  GE_ASSERT_NOTNULL(first_peer);
  auto first_consumer = std::dynamic_pointer_cast<af::AscNode>(first_peer->GetOwnerNode());
  GE_ASSERT_NOTNULL(first_consumer);
  GE_ASSERT_SUCCESS(InsertWorkspaceBoundary(graph, indirect_load, 0UL, first_consumer,
                                            static_cast<size_t>(first_peer->GetIdx()), boundary_name, false, false));
  if (il_output_anchors.size() > 1UL) {
    const std::string workspace_name = boundary_name + "_workspace";
    std::vector<af::InDataAnchorPtr> extra_peers;
    for (const auto &peer_input_anchor : il_output_anchors) {
      if (peer_input_anchor != first_peer && peer_input_anchor != nullptr) {
        extra_peers.emplace_back(peer_input_anchor);
      }
    }
    for (const auto &peer_input_anchor : extra_peers) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveEdge(output_anchor, peer_input_anchor));
    }
    size_t load_seq = 1UL;
    for (const auto &peer_input_anchor : extra_peers) {
      af::ascir_op::Workspace workspace_post(workspace_name.c_str());
      af::ascir_op::Load load((boundary_name + "_load" + std::to_string(load_seq)).c_str());
      const auto post_node = graph.AddNode(workspace_post);
      const auto load_node = graph.AddNode(load);
      GE_ASSERT_NOTNULL(post_node);
      GE_ASSERT_NOTNULL(load_node);
      // load 拷贝产出侧视图。post 与 InsertWorkspaceBoundary 保持一致：不拷贝
      // （Workspace 节点无视图/tensor_id 语义，拷贝源 tensor_id 会污染 workspace
      // 大小收集体系，导致各段偏移分配退化为 0）。
      GE_ASSERT_SUCCESS(CopyBoundaryTensorAttr(indirect_load, 0UL, load_node));
      GE_ASSERT_GRAPH_SUCCESS(
          af::GraphUtils::AddEdge(post_node->GetOutDataAnchor(0UL), load_node->GetInDataAnchor(0UL)));
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(load_node->GetOutDataAnchor(0UL), peer_input_anchor));
      ++load_seq;
    }
  }
  return af::SUCCESS;
}

af::Status BuildSkPartitionOrder(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load,
                                 std::vector<af::AscNodePtr> &node_order) {
  std::set<af::NodePtr> ordered_nodes;
  for (const char *role : {"input", "index", "output"}) {
    const std::string workspace_name = indirect_load->GetName() + "_sk_" + role + "_workspace";
    af::AscNodePtr workspace_pre;
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetName() == workspace_name && node->GetOutDataNodes().empty()) {
        workspace_pre = node;
        break;
      }
    }
    if (workspace_pre == nullptr) {
      // [SK分段SIMT] 窗口超限形态仅保留 output 边界（input/index 由 IL 段 GM 直读），
      // 缺失的边界 workspace 跳过而非报错。
      GELOGD("[IndirectLoad] SK partition order skips missing workspace[%s].", workspace_name.c_str());
      continue;
    }
    node_order.emplace_back(workspace_pre);
    ordered_nodes.emplace(workspace_pre);
  }
  std::vector<af::AscNodePtr> remaining_outputs;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetOutDataNodes().empty() && ordered_nodes.find(node) == ordered_nodes.end()) {
      remaining_outputs.emplace_back(node);
    }
  }
  std::sort(remaining_outputs.begin(), remaining_outputs.end(),
            [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) {
              return lhs->GetOpDescBarePtr()->GetId() < rhs->GetOpDescBarePtr()->GetId();
            });
  node_order.insert(node_order.end(), remaining_outputs.begin(), remaining_outputs.end());
  return af::SUCCESS;
}

af::Status ApplySkGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, bool &is_candidate_legal) {
  is_candidate_legal = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      GELOGI("[IndirectLoad] Skip SK candidate: Transpose is only supported by SIMD/SIMT.");
      return af::SUCCESS;
    }
  }
  RewrittenGraphAnalysis analysis;
  GE_ASSERT_SUCCESS(
      AnalyzeRewrittenGraph(graph, indirect_load, ascir::TemplateId::kIndirectLoadSK, is_candidate_legal, analysis));
  if (!is_candidate_legal) {
    GELOGI("[IndirectLoad] Reject SK: rewritten graph analysis is illegal.");
    return af::SUCCESS;
  }
  // SK 分区后消费子图走通用调度，多 Reduce 复合区域（如 LayerNorm 的
  // mean/sum 链）在通用路径会生成 R 轴切分模板，产生跨循环变量引用等
  // 未支持形态；暂时限制 SK 仅支持单 Reduce 场景，复合区域由
  // SIMD/SIMT/VectorFunc 路径处理，待 SK 消费子图调度完善后再放开。
  if (!analysis.composite_reduces.empty()) {
    is_candidate_legal = false;  // AnalyzeRewrittenGraph 已置 true，拒绝时必须显式复位
    GELOGI("[IndirectLoad] Reject SK: composite Norm region with %zu Reduce stages is not supported yet.",
           analysis.composite_reduces.size());
    return af::SUCCESS;
  }
  // SK 虽然采用 workspace 分区，但 Norm 区域的 Reduce/Broadcast 对应关系和
  // 轴保持证明仍必须与 SIMD/SIMT 共用，不能因模板不同而跳过语义校验。
  GE_ASSERT_SUCCESS(
      CollectGatherNormInfo(indirect_load, analysis, is_candidate_legal, ascir::TemplateId::kIndirectLoadSK));
  if (!is_candidate_legal) {
    GELOGI("[IndirectLoad] Reject SK: composite Norm validation is illegal.");
    return af::SUCCESS;
  }
  is_candidate_legal = false;
  if (!IsSkTemplateCandidateLegal(indirect_load)) {
    GELOGI("[IndirectLoad] Reject SK candidate for node[%s]: candidate legality check failed.",
           indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  is_candidate_legal = true;
  GE_ASSERT_SUCCESS(::ascir::SetTemplateId(indirect_load, ascir::TemplateId::kIndirectLoadSK));
  GE_ASSERT_SUCCESS(PartitionSkGraph(graph, indirect_load, analysis.align_input_path, analysis.align_index_path));
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  // axis=0 时轴前缀为空，boundary=0 会让 BuildAxisViewByBoundary 生成合成单轴
  // indirect_load_single_outer(size=1)，IL 的输入/输出整体落入 inner 全量视图，
  // 静态全量驻留估算超平台 UB，SK 被 StaticUbTemplateFilter 淘汰。
  // 与 SK 分段 SIMT 一致（RestoreSkTemplateAxes 注释），退化时改用输出 rank：
  // outer=完整输出视图，调度期按 UB 求解 tile。
  const size_t sk_boundary = axis_index == 0UL ? indirect_load->outputs()[0]->attr.axis.size() : axis_index;
  ascir::AxisId input_inner_axis = af::kIdNone;
  GE_ASSERT_SUCCESS(BuildSkInputInnerAxis(graph, indirect_load, axis_index, input_inner_axis));
  GE_ASSERT_SUCCESS(NormalizeAxesForTemplate(graph, indirect_load, sk_boundary, ascir::TemplateId::kIndirectLoadSK,
                                             input_inner_axis, af::kIdNone));
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  GE_ASSERT_TRUE(input_boundary != nullptr && af::ops::IsOps<af::ascir_op::Load>(input_boundary),
                 "IndirectLoad SK input boundary must be a Load node, node[%s].", indirect_load->GetNamePtr());
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
      input_boundary, ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary));
  return af::SUCCESS;
}

// [SK段SIMT化] 重定向的形态约束：IL 输出的下游闭包（DAG，LayerNorm 的
// x-mean 为菱形双消费）全部由 Cast/Reduce/Broadcast/Elementwise/Scalar/Store/
// Output 构成、且至少一个 Reduce 时允许重定向——即 Norm 类复合区域结构本身。
// 这是 R 非全载需要分段归约的目标形态：IL 段 SIMT gather 写 output workspace，
// 消费子图（单/多 Reduce 的 Norm DAG）独立调度，R 轴分 tile 由其自身 tiling 承载。
// 闭包含 IL/Transpose 等特殊算子或控制边时维持原 SK 行为不重定向。
bool IsNormRegionDag(const af::AscNodePtr &indirect_load) {
  std::vector<af::AscNodePtr> pending;
  std::unordered_set<const af::AscNode *> visited;
  int32_t reduce_count = 0;
  bool seen_store = false;
  // 起点为 IL 本身（不在白名单），仅检查其下游闭包。
  for (const auto &consumer : indirect_load->GetOutDataNodes()) {
    pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
  }
  for (size_t cursor = 0UL; cursor < pending.size() && cursor < 256UL; ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || !visited.emplace(node.get()).second) {
      continue;
    }
    const bool is_allowed = af::ops::IsOps<af::ascir_op::Cast>(node) || ScheduleUtils::IsReduce(node) ||
                            af::ops::IsOps<af::ascir_op::Broadcast>(node) ||
                            af::ops::IsOps<af::ascir_op::Store>(node) || af::ops::IsOps<af::ascir_op::Output>(node) ||
                            af::ops::IsOps<af::ascir_op::Scalar>(node) ||
                            af::ops::IsOps<af::ascir_op::ScalarData>(node) || ScheduleUtils::IsElewise(node);
    if (!is_allowed) {
      return false;
    }
    if (ScheduleUtils::IsReduce(node)) {
      ++reduce_count;
    }
    if (af::ops::IsOps<af::ascir_op::Store>(node)) {
      seen_store = true;
    }
    for (const auto &consumer : node->GetOutDataNodes()) {
      pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
    }
  }
  // [SK段SIMT化] 多 Reduce 复合区域（LayerNorm mean/var）的消费子图在通用调度
  // 生成 R 轴切分模板后，VF 节点跨循环引用统计链 tensor（codegen
  // ConnectApiCallInputs 找不到生产 ApiCall，实测 id[%d] 缺失），为通用 codegen
  // 的结构性缺口——维持单 Reduce 限制；多 Reduce 分段需消费子图内部再分段
  // （workspace 链），作为后续专项。
  return reduce_count >= 1 && seen_store;
}

// [SK段SIMT化·复合Norm] Norm 链分段化。通用调度对含 Reduce 的连通图按 R 轴切
// loop，Reduce/多消费者节点输出的下游若与其同分量，跨 loop 引用（codegen
// ConnectApiCallInputs 在单 loop 的 tensor_calls 中找不到生产者）。且连通性分区
// 为双向遍历（上下游都走），LayerNorm 的 x-mean 菱形（diff 被方差链与输出链双
// 消费）使单一 Reduce 边界无法切断分量。规则：对每个 Reduce 节点、每个多消费
// 者计算节点的输出，逐消费者独立插 workspace 边界（每边一份，彻底切断连通分
// 量）——每个子图数据流自足（输入均为 Load），R 切分不再产生跨循环依赖，与独
// 立 LayerNorm 的统计段/消费段分离形态对齐。单消费者非 Reduce 节点不切（同段
// 内常规流水）。
af::Status PartitionNormReduceBoundaries(af::AscGraph &graph, const af::AscNodePtr &indirect_load) {
  // [边界化范围] 仅 IL 输出下游闭包（Norm DAG）：输入前置链（Scalar 标量广播等
  // side-input 生产链）不在范围内——对它边界化会使 SIMT 流程的输入回溯
  // （CompleteInputDataTensorAttrs：Load 的 producer 必须是 Data）遇到边界
  // load（producer 为 workspace）而失败（生产 9.28：scalar1 多消费者被误切）。
  std::unordered_set<const af::AscNode *> downstream;
  {
    std::vector<af::AscNodePtr> pending;
    for (const auto &consumer : indirect_load->GetOutDataNodes()) {
      pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
    }
    for (size_t cursor = 0UL; cursor < pending.size() && cursor < 256UL; ++cursor) {
      const af::AscNodePtr node = pending[cursor];
      if (node == nullptr || !downstream.emplace(node.get()).second) {
        continue;
      }
      for (const auto &consumer : node->GetOutDataNodes()) {
        pending.emplace_back(std::dynamic_pointer_cast<af::AscNode>(consumer));
      }
    }
  }
  std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> boundary_targets;
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || node == indirect_load || downstream.find(node.get()) == downstream.end()) {
      continue;  // 仅 IL 下游闭包内的节点；IL 输出由 PartitionSkGraph 的 output 边界处理
    }
    const bool is_boundary_node =
        af::ops::IsOps<af::ascir_op::Workspace>(node) || af::ops::IsOps<af::ascir_op::Store>(node) ||
        af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Data>(node) ||
        af::ops::IsOps<af::ascir_op::Output>(node);
    if (is_boundary_node) {
      continue;
    }
    const auto out_anchor = node->GetOutDataAnchor(0UL);
    if (out_anchor == nullptr) {
      continue;
    }
    const auto &peers = out_anchor->GetPeerInDataAnchors();
    if (peers.empty()) {
      continue;
    }
    bool has_non_store_consumer = false;
    for (const auto &peer : peers) {
      const auto consumer = peer != nullptr && peer->GetOwnerNode() != nullptr
                                ? std::dynamic_pointer_cast<af::AscNode>(peer->GetOwnerNode())
                                : nullptr;
      if (consumer != nullptr && !af::ops::IsOps<af::ascir_op::Store>(consumer)) {
        has_non_store_consumer = true;
        break;
      }
    }
    if (!has_non_store_consumer) {
      continue;  // 输出直连 Store（段尾）
    }
    if (ScheduleUtils::IsReduce(node) || peers.size() > 1UL) {
      for (const auto &peer : peers) {
        const auto consumer = peer != nullptr && peer->GetOwnerNode() != nullptr
                                  ? std::dynamic_pointer_cast<af::AscNode>(peer->GetOwnerNode())
                                  : nullptr;
        if (consumer != nullptr) {
          boundary_targets.emplace_back(node, consumer);
        }
      }
    }
  }
  size_t boundary_seq = 0UL;
  for (const auto &target : boundary_targets) {
    const auto &node = target.first;
    const auto &consumer = target.second;
    const auto out_anchor = node->GetOutDataAnchor(0UL);
    // 按消费者定位其当前连到本生产者输出的输入边（边被逐条改接，重新查找）。
    af::InDataAnchorPtr peer_anchor = nullptr;
    const auto consumer_inputs = consumer->GetInDataNodes();
    for (size_t in_idx = 0UL; in_idx < consumer_inputs.size(); ++in_idx) {
      const auto in_anchor = consumer->GetInDataAnchor(in_idx);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() == out_anchor) {
        peer_anchor = in_anchor;
        break;
      }
    }
    if (peer_anchor == nullptr) {
      continue;  // 该边已被此前的边界化改接
    }
    const std::string boundary_name = node->GetName() + "_sk_n" + std::to_string(boundary_seq);
    GE_ASSERT_SUCCESS(InsertWorkspaceBoundary(graph, node, 0UL, consumer, static_cast<size_t>(peer_anchor->GetIdx()),
                                              boundary_name, false, false));
    ++boundary_seq;
    GELOGI("[IndirectLoad] SK norm boundary[%s] inserted (node[%s] consumer[%s]).", boundary_name.c_str(),
           node->GetNamePtr(), consumer->GetNamePtr());
  }
  return af::SUCCESS;
}
}  // namespace indirect_load_detail
}  // namespace optimize
