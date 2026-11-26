/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

bool IsSupportedIndirectLoadTopologyNode(const af::AscNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (ScheduleUtils::IsIOBuffer(node) || ScheduleUtils::IsBuffer(node) ||
      af::ops::IsOps<af::ascir_op::IndirectLoad>(node)) {
    return true;
  }
  if (GetLoadTransposeSource(node) != nullptr) {
    return true;
  }
  return ScheduleUtils::IsLoad(node) || ScheduleUtils::IsStore(node) || ScheduleUtils::IsElewise(node) ||
         ScheduleUtils::IsBroadcast(node) || ScheduleUtils::IsReduce(node);
}

void CollectInputRegionMembers(const af::AscNodePtr &indirect_load, size_t input_index, NodeSet &region) {
  const af::AscNodePtr root = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
  if (root == nullptr) {
    return;
  }
  TraverseInputProducers({root}, region,
                         [](const af::AscNodePtr &node) { return !af::ops::IsOps<af::ascir_op::Load>(node); });
}

bool CollectSimtBackwardRegion(const NodePath &roots, const af::AscNodePtr &indirect_load, NodeSet &region) {
  NodePath pending = roots;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr) {
      return false;
    }
    if (node == indirect_load || IsInputDataSource(node)) {
      continue;
    }
    if (!region.emplace(node.get()).second || af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    if (node->inputs.Size() == 0UL && !af::ops::IsOps<af::ascir_op::Arange>(node)) {
      return false;
    }
    if (af::ops::IsOps<af::ascir_op::Arange>(node)) {
      continue;
    }
    for (size_t i = 0UL; i < node->inputs.Size(); ++i) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(node, i);
      if (producer == nullptr) {
        return false;
      }
      pending.emplace_back(producer);
    }
  }
  return true;
}

void CollectSimtOutputConsumerClosure(const af::AscNodePtr &indirect_load, NodeSet &closure) {
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    if (consumer != nullptr) {
      roots.emplace_back(consumer);
    }
  }
  const auto visit = [](const af::AscNodePtr &node, bool &stop) -> af::Status {
    (void)node;
    (void)stop;
    return af::SUCCESS;
  };
  (void)TraverseOutputConsumers(roots, closure, visit);
}

// [index 链外部 store] index 生产链尾部张量的区域外直连 store 会被 lowering 收为
// 额外输出链（链值 = Index() 求值器结果），此处将其纳入 SIMT 融合区域（获得
// kSimtDirectGmBoundary 角色，跳过普通调度发射）；index 链其余无法供值的外部消费者
// 静默丢弃会导致目标 buffer 永不写入，必须整候选淘汰。合法消费者集 = IndirectLoad
// 自身 ∪ 融合区域（index 链与输出链的 backward 闭包——含经多输入算子合流进输出链
// 的 side-input，如 embedding+reduce 图中消费 index load 的 Ge/Select）∪ IL 输出的
// 前向闭包（post-Reduce 普通调度链）；守卫只扫 index 链子集，不扫全融合区域。
bool FoldSimtIndexChainExternalStores(const af::AscNodePtr &indirect_load, const af::AscNodePtr &index_root,
                                      NodeSet &region, const NodeSet &group_exempt, const NodeSet *pre_index_chain) {
  NodeSet index_chain;
  if (pre_index_chain != nullptr) {
    index_chain = *pre_index_chain;
  } else if (!CollectSimtBackwardRegion({index_root}, indirect_load, index_chain)) {
    return false;
  }
  NodeSet output_consumers;
  CollectSimtOutputConsumerClosure(indirect_load, output_consumers);
  const auto owner_graph = indirect_load->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    return false;
  }
  const auto inputs = indirect_load->inputs();
  if (inputs.size() <= ascgen_utils::indirect_load::kIndexTensorIndex) {
    return false;
  }
  const ascir::TensorId index_tensor_id = inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr.mem.tensor_id;
  for (const auto &graph_node : owner_graph->GetDirectNode()) {
    const auto node = std::dynamic_pointer_cast<af::AscNode>(graph_node);
    if (node == nullptr || index_chain.count(node.get()) == 0UL) {
      continue;
    }
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (consumer == nullptr || consumer == indirect_load || region.count(consumer.get()) != 0UL ||
          output_consumers.count(consumer.get()) != 0UL) {
        continue;
      }
      // [跨成员豁免] 消费者在同组另一成员的 SIMT index 链里——由该成员的 evaluator
      // 逐元素消费（寄存器级），不要求向量侧物化 tensor，不是真正外部消费者。
      if (group_exempt.count(consumer.get()) != 0UL) {
        continue;
      }
      if (node == index_root && af::ops::IsOps<af::ascir_op::Store>(consumer) && !consumer->inputs().empty() &&
          consumer->inputs()[0]->attr.mem.tensor_id == index_tensor_id) {
        region.emplace(consumer.get());
        continue;
      }
      GELOGI("[IndirectLoad] Reject SIMT candidate: index chain node[%s] has unsupported external consumer[%s, %s].",
             node->GetNamePtr(), consumer->GetNamePtr(), consumer->GetTypePtr());
      return false;
    }
  }
  return true;
}

void CollectSimtFusedRegionMembers(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                   NodeSet &region) {
  region.clear();
  const af::AscNodePtr index_root =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex);
  const af::AscNodePtr output_terminal =
      analysis.post_reduce == nullptr ? analysis.output_store
                                      : ascgen_utils::indirect_load::GetSimtPostReduceLocalProducer(indirect_load);
  if (index_root == nullptr || output_terminal == nullptr) {
    return;
  }
  if (!CollectSimtBackwardRegion({index_root, output_terminal}, indirect_load, region)) {
    region.clear();
    return;
  }
  if (!FoldSimtIndexChainExternalStores(indirect_load, index_root, region, NodeSet{})) {
    region.clear();
  }
}

// The IndirectLoad logical view is expressed in the Transpose output order,
// while the direct-GM boundary still reads the original Load buffer. Map the
// source Load strides into that output order for the strided address policy.
af::Status BuildLoadTransposeSourceView(const af::AscNodePtr &transpose, const af::AscTensorAttr &logical_attr,
                                        ascgen_utils::indirect_load::LogicalTensorView &view) {
  const auto load = ascgen_utils::indirect_load::GetInputProducer(transpose, 0UL);
  const auto &source = load->outputs[0].attr;
  GE_ASSERT_TRUE(source.axis.size() == source.strides.size() &&
                     logical_attr.axis.size() == logical_attr.repeats.size() &&
                     logical_attr.axis.size() == logical_attr.strides.size(),
                 "IndirectLoad Load->Transpose tensor metadata rank mismatch.");
  view = {logical_attr.axis, logical_attr.repeats, logical_attr.strides};
  for (size_t output_dim = 0UL; output_dim < logical_attr.axis.size(); ++output_dim) {
    const auto source_axis = std::find(source.axis.begin(), source.axis.end(), logical_attr.axis[output_dim]);
    GE_ASSERT_TRUE(source_axis != source.axis.end(), "IndirectLoad Transpose output axis is absent from Load source.");
    const size_t source_dim = static_cast<size_t>(std::distance(source.axis.begin(), source_axis));
    view.strides[output_dim] = source.strides[source_dim];
  }
  return af::SUCCESS;
}

bool CanMoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &consumer) {
  if (node == nullptr) {
    return false;
  }
  // Data and Load terminate the movable input-pre chain without being moved.
  if (af::ops::IsOps<af::ascir_op::Data>(node) || af::ops::IsOps<af::ascir_op::Load>(node)) {
    return false;
  }
  // Branching, control dependencies, and non-unary nodes cannot be safely reordered across IndirectLoad.
  if (node->inputs.Size() != 1UL || HasControlEdge(node) ||
      ascgen_utils::indirect_load::GetOnlyOutputConsumer(node) != consumer) {
    return false;
  }
  // Ordinary unary elementwise nodes can be moved; other compute types remain on the input side.
  return ScheduleUtils::IsElewise(node);
}

bool IsSimdInputShapeEnlarged(const af::AscNodePtr &indirect_load) {
  af::Expression input_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat :
       indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats) {
    input_numel = input_numel * repeat;
  }
  af::Expression output_numel = af::sym::kSymbolOne;
  for (const af::Expression &repeat : indirect_load->outputs()[0]->attr.repeats) {
    output_numel = output_numel * repeat;
  }
  return af::SymbolicUtils::StaticCheckGt(output_numel, input_numel) == af::TriBool::kTrue;
}

af::Status MoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &indirect_load) {
  const auto node_in = node->GetInDataAnchor(0UL);
  const auto indirect_in = indirect_load->GetInDataAnchor(ascgen_utils::indirect_load::kInputTensorIndex);
  GE_ASSERT_NOTNULL(node_in);
  GE_ASSERT_NOTNULL(indirect_in);
  const auto producer_out = node_in->GetPeerOutAnchor();
  const auto node_out = indirect_in->GetPeerOutAnchor();
  const auto indirect_out = indirect_load->GetOutDataAnchor(0UL);
  GE_ASSERT_NOTNULL(producer_out);
  GE_ASSERT_NOTNULL(node_out);
  GE_ASSERT_NOTNULL(indirect_out);
  const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_out->GetOwnerNode());
  GE_ASSERT_NOTNULL(producer);
  GE_ASSERT_TRUE(node_out->GetOwnerNode() == node, "IndirectLoad input-pre edge does not match node[%s].",
                 node->GetNamePtr());
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(node_out, indirect_in, producer_out));
  GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(producer_out, node_in, indirect_out));
  const auto peer_inputs = indirect_out->GetPeerInDataAnchors();
  for (const auto &peer_in : peer_inputs) {
    if (peer_in != node_in) {
      GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::ReplaceEdgeSrc(indirect_out, peer_in, node_out));
    }
  }

  const auto indirect_output = indirect_load->outputs().front();
  const size_t producer_output_index = static_cast<size_t>(producer_out->GetIdx());
  const size_t node_output_index = static_cast<size_t>(node_out->GetIdx());
  GE_ASSERT_TRUE(producer_output_index < producer->outputs().size(),
                 "IndirectLoad input-pre producer output index is invalid.");
  GE_ASSERT_TRUE(node_output_index < node->outputs().size(), "IndirectLoad input-pre output index is invalid.");
  const auto node_output = node->outputs()[node_output_index];
  node->attr.sched.axis = indirect_load->attr.sched.axis;
  node_output->attr.axis = indirect_output->attr.axis;
  node_output->attr.repeats = indirect_output->attr.repeats;
  node_output->attr.strides = indirect_output->attr.strides;
  indirect_output->attr.dtype = producer->outputs()[producer_output_index]->attr.dtype;
  GELOGD("[IndirectLoad] Move input pre node[%s] after IndirectLoad[%s].", node->GetNamePtr(),
         indirect_load->GetNamePtr());
  return af::SUCCESS;
}

af::Status RewriteInputPreNodes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return af::SUCCESS;
  }
  if (HasControlEdge(indirect_load)) {
    GELOGI("[IndirectLoad] Skip moving input-pre nodes for node[%s].", indirect_load->GetNamePtr());
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimd && IsSimdInputShapeEnlarged(indirect_load)) {
    const auto &input_shape = indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.repeats;
    const auto &output_shape = indirect_load->outputs()[0]->attr.repeats;
    GELOGI("[IndirectLoad] Skip moving SIMD input-pre nodes for node[%s]: input shape[%s], output shape[%s].",
           indirect_load->GetNamePtr(), af::ToString(input_shape).c_str(), af::ToString(output_shape).c_str());
    return af::SUCCESS;
  }
  // Keep a direct Load->Transpose boundary on the SIMT input side.  Its
  // permutation is represented by the SIMT address policy; moving it across
  // IndirectLoad would discard the source physical view.
  const auto input_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  if (template_id == ascir::TemplateId::kIndirectLoadSimt && GetLoadTransposeSource(input_producer) != nullptr) {
    return af::SUCCESS;
  }
  NodePath movable_nodes;
  af::AscNodePtr consumer = indirect_load;
  for (af::AscNodePtr node =
           ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
       CanMoveInputPreNode(node, consumer); node = ascgen_utils::indirect_load::GetInputProducer(node, 0UL)) {
    movable_nodes.emplace_back(node);
    consumer = node;
  }
  if (movable_nodes.empty()) {
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : movable_nodes) {
    GE_ASSERT_SUCCESS(MoveInputPreNode(node, indirect_load));
  }
  return ScheduleUtils::TopologicalSorting(graph);
}

af::Status RewriteSimtInputBroadcast(const af::AscNodePtr &indirect_load, InputViewPlan &input_plan,
                                     bool &is_candidate_legal) {
  const af::AscNodePtr input_producer =
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  if (!IsBroadcastNode(input_producer)) {
    return af::SUCCESS;
  }

  const auto broadcast_iter = std::find(input_plan.path.begin(), input_plan.path.end(), input_producer);
  GE_ASSERT_TRUE(broadcast_iter != input_plan.path.end(),
                 "SIMT direct input Broadcast is missing from collected path.");
  input_plan.path_broadcast_index = static_cast<int64_t>(std::distance(input_plan.path.begin(), broadcast_iter));
  GE_ASSERT_SUCCESS(FoldBroadcastPath(input_plan, ascir::TemplateId::kIndirectLoadSimt, is_candidate_legal));
  return af::SUCCESS;
}

// 不区分一元/二元路径，沿所有上游数据边收集 Load；Load 节点是输入边界，停止继续回溯。
void CollectReachableLoads(const af::AscNodePtr &root, NodeSet &visited, NodePath &loads) {
  if (root == nullptr) {
    return;
  }
  TraverseInputProducers({root}, visited, [&loads](const af::AscNodePtr &node) {
    if (af::ops::IsOps<af::ascir_op::Load>(node)) {
      loads.emplace_back(node);
      return false;
    }
    return true;
  });
}

af::Status CompleteInputDataTensorAttrs(const RewrittenGraphAnalysis &analysis) {
  // 仅对可达 Load 补齐对应 Data；没有 Load 的路径无需处理。
  NodeSet visited;
  for (const af::AscNodePtr &root : {analysis.input_root, analysis.index_root}) {
    NodePath loads;
    CollectReachableLoads(root, visited, loads);
    for (const af::AscNodePtr &load : loads) {
      const af::AscNodePtr data = ascgen_utils::indirect_load::GetInputProducer(load, 0UL);
      GE_ASSERT_NOTNULL(data, "IndirectLoad input Load[%s] has no producer.", load->GetNamePtr());
      GE_ASSERT_TRUE(af::ops::IsOps<af::ascir_op::Data>(data), "IndirectLoad input Load[%s] producer is not Data.",
                     load->GetNamePtr());
      const auto load_outputs = load->outputs();
      const auto data_outputs = data->outputs();
      GE_ASSERT_TRUE(!load_outputs.empty() && !data_outputs.empty(),
                     "IndirectLoad input Load[%s] or Data[%s] has no output.", load->GetNamePtr(), data->GetNamePtr());
      const auto load_out = load_outputs.front();
      const auto data_out = data_outputs.front();
      GE_ASSERT_NOTNULL(load_out, "IndirectLoad input Load[%s] output is null.", load->GetNamePtr());
      GE_ASSERT_NOTNULL(data_out, "IndirectLoad input Data[%s] output is null.", data->GetNamePtr());
      GE_ASSERT_TRUE(!load_out->attr.axis.empty() && load_out->attr.axis.size() == load_out->attr.repeats.size() &&
                         load_out->attr.axis.size() == load_out->attr.strides.size(),
                     "IndirectLoad input Load[%s] tensor attributes are incomplete.", load->GetNamePtr());
      GELOGD("[IndirectLoad] Complete tensor attrs from Load[%s] to Data[%s].", load->GetNamePtr(), data->GetNamePtr());
      data_out->attr.axis = load_out->attr.axis;
      data_out->attr.repeats = load_out->attr.repeats;
      data_out->attr.strides = load_out->attr.strides;
      data_out->attr.dtype = load_out->attr.dtype;
    }
  }
  return af::SUCCESS;
}

// 收集 IndirectLoad 两个输入的路径，并按模板规则选出待处理的 Broadcast：SIMD 只看直接生产者，SIMT/SK
// 沿单输入链回溯并查找最近 Broadcast。
void CollectInputPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                       PhysicalViewPreparation &preparation) {
  const bool collect_full_path = template_id != ascir::TemplateId::kIndirectLoadSimd;
  for (size_t input_index :
       {ascgen_utils::indirect_load::kInputTensorIndex, ascgen_utils::indirect_load::kIndexTensorIndex}) {
    InputViewPlan &plan =
        input_index == ascgen_utils::indirect_load::kInputTensorIndex ? preparation.input : preparation.index;
    NodePath &path = plan.path;
    plan.path_broadcast_index = kInvalidBroadcastIndex;
    for (af::AscNodePtr current = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_index);
         current != nullptr; current = collect_full_path && current->inputs.Size() == 1UL
                                           ? ascgen_utils::indirect_load::GetInputProducer(current, 0UL)
                                           : nullptr) {
      if (plan.path_broadcast_index == kInvalidBroadcastIndex && IsBroadcastNode(current)) {
        plan.path_broadcast_index = static_cast<int64_t>(path.size());
      }
      path.emplace_back(current);
    }
  }
}

af::Status ResolveInputPhysicalView(const af::AscTensorAttr &logical_attr, ascir::TemplateId template_id,
                                    InputViewPlan &plan, af::AscTensorAttr &physical_attr, bool &is_supported) {
  physical_attr = logical_attr;
  plan.load_transpose = nullptr;
  const bool has_broadcast = plan.path_broadcast_index != kInvalidBroadcastIndex;
  af::AscNodePtr source = plan.path.empty() ? nullptr : plan.path.front();
  if (has_broadcast) {
    const auto broadcast = plan.path[static_cast<size_t>(plan.path_broadcast_index)];
    source = ascgen_utils::indirect_load::GetInputProducer(broadcast, 0UL);
    if (source == nullptr) {
      GELOGI("[IndirectLoad] Reject candidate[%d]: Broadcast node[%s] source is invalid.",
             static_cast<int32_t>(template_id), broadcast->GetNamePtr());
      is_supported = false;
      return af::SUCCESS;
    }
    GE_ASSERT_SUCCESS(GetBroadcastPhysicalAttr(broadcast, template_id, physical_attr));
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK || GetLoadTransposeSource(source) == nullptr) {
    return af::SUCCESS;
  }
  // SIMD may use the source Load only when the intervening Transpose can be folded safely.
  // SIMT already reads GM directly, so it only needs the axis-mapped view, not a graph rewrite.
  if (has_broadcast && template_id == ascir::TemplateId::kIndirectLoadSimd &&
      GetFoldableLoadTransposeSource(source) == nullptr) {
    return af::SUCCESS;
  }
  plan.load_transpose = source;
  ascgen_utils::indirect_load::LogicalTensorView source_view;
  GE_ASSERT_SUCCESS(BuildLoadTransposeSourceView(source, physical_attr, source_view));
  physical_attr.strides = std::move(source_view.strides);
  return af::SUCCESS;
}

af::Status AnalyzeInputPath(const af::AscNodePtr &indirect_load, size_t input_idx, ascir::TemplateId template_id,
                            InputViewPlan &plan, bool &is_path_supported) {
  is_path_supported = true;
  const auto inputs = indirect_load->inputs();
  const auto &logical_attr = inputs[input_idx]->attr;
  af::AscTensorAttr physical_attr;
  GE_ASSERT_SUCCESS(ResolveInputPhysicalView(logical_attr, template_id, plan, physical_attr, is_path_supported));
  if (!is_path_supported) {
    return af::SUCCESS;
  }
  const bool has_broadcast = plan.path_broadcast_index != kInvalidBroadcastIndex;
  ascgen_utils::indirect_load::LogicalTensorView view{logical_attr.axis, logical_attr.repeats, physical_attr.strides};
  if (has_broadcast) {
    GE_ASSERT_SUCCESS(BuildBroadcastLogicalView(logical_attr, physical_attr, template_id, view));
  }
  plan.simd_index_uses_output_inner_axis = template_id == ascir::TemplateId::kIndirectLoadSimd &&
                                           input_idx == ascgen_utils::indirect_load::kIndexTensorIndex &&
                                           (has_broadcast || plan.load_transpose != nullptr);
  if (plan.load_transpose != nullptr) {
    // Permutations are valid non-overlapping strided views even when strides are not monotonically decreasing.
    plan.layout = {{view.axis_ids, view.sizes, view.strides},
                   ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided,
                   physical_attr.repeats};
  } else {
    // 转置稠密+零步长广播轴的混合布局仅对 SIMT 放行：SIMT 直读 GM 按 strides 寻址
    // （置换视图片 kStrided 寻址既有支持），零步长轴等价于重复读同一物理元素；SIMD 的
    // UB staging 与 SK workspace 的 store 段按行主序批量搬运，不接受该混合布局。
    // 既有"单调+零步长"容忍分支（allow_non_overlapping_zero_stride）语义不变，SK 照旧。
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(
        view, plan.layout,
        (template_id == ascir::TemplateId::kIndirectLoadSimt || template_id == ascir::TemplateId::kIndirectLoadSK) &&
            has_broadcast,
        template_id == ascir::TemplateId::kIndirectLoadSimt && has_broadcast));
    if (template_id == ascir::TemplateId::kIndirectLoadSK && has_broadcast &&
        plan.layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
      plan.layout.physical_repeats = physical_attr.repeats;
    }
  }
  if (plan.layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported) {
    GELOGI("[IndirectLoad] Reject candidate[%d]: input path layout%s is unsupported.",
           static_cast<int32_t>(template_id), has_broadcast ? " with Broadcast source view" : "");
    is_path_supported = false;
  }
  return af::SUCCESS;
}

af::Status RewriteSkInputPaths(PhysicalViewPreparation &preparation, RewrittenGraphAnalysis &analysis,
                               bool &is_candidate_legal) {
  for (InputViewPlan *const plan : {&preparation.input, &preparation.index}) {
    GE_ASSERT_SUCCESS(FoldBroadcastPath(*plan, ascir::TemplateId::kIndirectLoadSK, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }
  analysis.input_path = std::move(preparation.input.path);
  analysis.index_path = std::move(preparation.index.path);
  return af::SUCCESS;
}

af::Status RewriteSimdInputLayouts(PhysicalViewPreparation &preparation) {
  for (InputViewPlan *plan : {&preparation.input, &preparation.index}) {
    if (plan->path_broadcast_index == kInvalidBroadcastIndex || plan->load_transpose == nullptr) {
      continue;
    }
    const auto broadcast = plan->path[static_cast<size_t>(plan->path_broadcast_index)];
    const auto &transpose = plan->load_transpose;
    const auto load = ascgen_utils::indirect_load::GetInputProducer(transpose, 0UL);
    GE_ASSERT_NOTNULL(load);
    const auto owner = transpose->GetOwnerComputeGraph();
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::IsolateNodeOneIO(transpose));
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::RemoveNodeWithoutRelink(owner, transpose));
    // Once the Transpose is removed, the Load and Broadcast share the resolved physical view.
    GE_ASSERT_SUCCESS(ApplyPhysicalView(NodePath{load, broadcast}, plan->layout));
  }
  return af::SUCCESS;
}

af::Status RewriteBroadcastPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                 PhysicalViewPreparation &preparation, RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal) {
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    return RewriteSkInputPaths(preparation, analysis, is_candidate_legal);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return RewriteSimtInputBroadcast(indirect_load, preparation.input, is_candidate_legal);
  }
  return RewriteSimdInputLayouts(preparation);
}

namespace {
// ir_attr.max 是 index 值域上界，通常为数据表 gather 轴尺寸，也可为宽松上界。
// 模板归一化重写输入视图后 axis 不随之重映射（rank-1 图 [vocab] 被改写为 [1, vocab]
// 后 vocab 落在 Dim1 而 axis 仍为 0，SIMT 以 strides[0]=vocab 做 gather 步长越界，
// 触发 aivec 334）。max 大于 axis 维尺寸且与唯一其它维相等时把 axis 重映射到该维；
// 无匹配（宽松上界）、多匹配或不可静态证明时保持既有行为。
af::Status RemapMisalignedGatherAxisIfSingular(const af::AscNodePtr &indirect_load,
                                               const ascgen_utils::indirect_load::TemplateLogicalView &logical_view) {
  size_t remapped_axis = 0UL;
  bool gather_extent_mismatch = false;
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  auto *il_ir_attr =
      indirect_load->attr.ir_attr == nullptr
          ? nullptr
          : indirect_load->attr.ir_attr->DownCastTo<af::ascir_op::IndirectLoad::AscIndirectLoadIrAttrDef>();
  af::Expression gather_extent;
  if (axis_index != kIndirectLoadInvalidAxisIndex && axis_index < logical_view.input.sizes.size() &&
      il_ir_attr != nullptr && il_ir_attr->GetMax(gather_extent) == af::SUCCESS && gather_extent.IsValid() &&
      af::SymbolicUtils::StaticCheckGt(gather_extent, logical_view.input.sizes[axis_index]) == af::TriBool::kTrue) {
    size_t match_count = 0UL;
    for (size_t dim = 0UL; dim < logical_view.input.sizes.size(); ++dim) {
      if (af::SymbolicUtils::StaticCheckEq(logical_view.input.sizes[dim], gather_extent) == af::TriBool::kTrue) {
        ++match_count;
        remapped_axis = dim;
      }
    }
    gather_extent_mismatch = (match_count == 1UL);
  }
  if (gather_extent_mismatch) {
    GELOGI("[IndirectLoad] Remap node[%s] axis %zu -> %zu: candidate view moved the gather dim.",
           indirect_load->GetNamePtr(), axis_index, remapped_axis);
    GE_ASSERT_SUCCESS(il_ir_attr->SetAxis(static_cast<int64_t>(remapped_axis)));
  }
  return af::SUCCESS;
}
}  // namespace

af::Status PreparePhysicalViews(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                PhysicalViewPreparation &preparation, bool &is_candidate_legal) {
  // 输入/输出个数已在 Generate 入口统一校验，axis 由 AnalyzeRewrittenGraph 统一获取，此处不再重复获取。
  is_candidate_legal = true;
  CollectInputPaths(indirect_load, template_id, preparation);
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    InputViewPlan &plan =
        input_idx == ascgen_utils::indirect_load::kInputTensorIndex ? preparation.input : preparation.index;
    bool is_path_supported = true;
    GE_ASSERT_SUCCESS(AnalyzeInputPath(indirect_load, input_idx, template_id, plan, is_path_supported));
    if (!is_path_supported) {
      is_candidate_legal = false;
      const char *const path_name = input_idx == ascgen_utils::indirect_load::kIndexTensorIndex ? "index" : "input";
      GELOGI("[IndirectLoad] Reject candidate[%d]: %s path is unsupported.", static_cast<int32_t>(template_id),
             path_name);
      return af::SUCCESS;
    }
  }
  preparation.logical_view.input = preparation.input.layout;
  preparation.logical_view.index = preparation.index.layout;
  const auto outputs = indirect_load->outputs();
  const auto &output_attr = outputs.front()->attr;
  preparation.logical_view.output = {output_attr.axis, output_attr.repeats, output_attr.strides};
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateIndirectLoadOutputLayout(preparation.logical_view.output));
  GE_ASSERT_SUCCESS(RemapMisalignedGatherAxisIfSingular(indirect_load, preparation.logical_view));
  ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::AnalyzeIndirectLoadAccess(indirect_load, preparation.logical_view, access_info));
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetIndirectLoadAccessInfo(indirect_load, access_info));
  // 本函数只做只读分析；Broadcast 删除与物理视图写回在 AnalyzeRewrittenGraph 中统一执行。
  return ascgen_utils::indirect_load::SetTemplateLogicalView(indirect_load, preparation.logical_view);
}

af::Status CollectRewrittenRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  ascir::TemplateId template_id, RewrittenGraphAnalysis &analysis) {
  NodeSet input_region_set;
  NodeSet index_region_set;
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex, input_region_set);
    CollectInputRegionMembers(indirect_load, ascgen_utils::indirect_load::kIndexTensorIndex, index_region_set);
  } else if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 的 index 链与输出链共享统一轴、下游统一处理，融合集整体归入 index_region；region 保持为空。
    CollectSimtFusedRegionMembers(indirect_load, analysis, index_region_set);
  } else {
    GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
  }
  // 一次 GetAllNodes 收集 region 与 index_region，保证顺序确定性。
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (index_region_set.count(node.get()) != 0UL) {
      analysis.index_region.emplace_back(node);
    }
    if (input_region_set.count(node.get()) != 0UL) {
      analysis.input_region.emplace_back(node);
    }
  }
  return af::SUCCESS;
}

af::Status ApplyTemplatePathLayouts(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                    const PhysicalViewPreparation &preparation,
                                    const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    // SIMT 融合集（index_region）从 index/output root 回溯，不包含 IndirectLoad 的 input
    // root，因此需要单独写回直接输入生产者。
    const InputViewPlan *const m2_plans[] = {&preparation.input, &preparation.index};
    for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx);
      if (producer != nullptr) {
        GE_ASSERT_SUCCESS(ApplyIndirectLoadPathLayout(NodePath{producer}, m2_plans[input_idx]->layout, false));
      }
    }
    return af::SUCCESS;
  }

  const bool is_simd = template_id == ascir::TemplateId::kIndirectLoadSimd;
  const NodePath &input_path = is_simd ? analysis.input_region : analysis.input_path;
  const NodePath &index_path = is_simd ? analysis.index_region : analysis.index_path;
  const auto &input_layout = preparation.input.layout;
  const auto &index_layout = preparation.index.layout;
  // SK 将输入链切分为独立的 GM->workspace 子图。对 Broadcast 源的 strided
  // view，workspace 边界会从该链末端的 Load 拷贝 tensor attr；若不在切分前
  // 写回物理 view，子图会把 Broadcast 后的逻辑 extent 当成源 GM 连续布局，
  // 导致错误的 GM 偏移。SIMD/SIMT 仍沿用各自的 strided 处理方式。
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    if (input_layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
      GE_ASSERT_SUCCESS(ApplyPhysicalView(input_path, input_layout));
    }
    if (index_layout.kind == ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided) {
      GE_ASSERT_SUCCESS(ApplyPhysicalView(index_path, index_layout));
    }
  }
  // [post-Reduce 稠密视图] IL 输出链到达 Reduce（含经 elementwise 中间链，如
  // gather+ele+sum / gather+norm）时不做 kStridedUbPath 对齐标注：该标注会经
  // SetVectorizedStridesForTensor(kAligned) 把视图尾轴按对齐块描述（如 4→8），
  // 而 SIMD 实际写出为稠密布局（only_gather 同形态实证），视图与实现不符导致
  // 下游 Abs/Reduce/DataCopy 按 stride8 错位访问（AIC 341）。无 post-Reduce 的
  // 纯 gather 输出直连 Store 场景维持原标注语义不变。
  const bool has_post_reduce = analysis.post_reduce != nullptr;
  GE_ASSERT_SUCCESS(
      ApplyIndirectLoadPathLayout(input_path, input_layout, analysis.align_input_path && !has_post_reduce));
  GE_ASSERT_SUCCESS(
      ApplyIndirectLoadPathLayout(index_path, index_layout, analysis.align_index_path && !has_post_reduce));
  return af::SUCCESS;
}

// 折叠与搬移互相暴露输入：Broadcast 折叠后 Cast 才成为直接生产者（Cast→Bc 交错），
// Cast 搬走后 Broadcast 才成为直接生产者（Bc→Cast 交错）。收敛保证：每个 Broadcast
// 最多被折一次（不可逆删节点）+ 节点只从 IL 上游单向移到下游（不可能回移）
// => 轮数 <= 链上 Broadcast 数 + 1。仅 SIMT 需要多轮（SIMD/SK 的折叠是一次性的，
// RewriteInputPreNodes 单次即可搬完可搬链）。命中上限且 boundary 仍非 Load 时 WARN。
af::Status RewriteInputChainToFixpoint(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                       ascir::TemplateId template_id, PhysicalViewPreparation &preparation,
                                       RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  constexpr size_t kMaxRewritePasses = 8UL;
  bool hit_cap = false;
  for (size_t pass = 0UL; pass < kMaxRewritePasses; ++pass) {
    const auto producer_before =
        ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
    // pass>0 时按当前图重新收集输入路径（pass0 已在 PreparePhysicalViews 收集过）。
    // load_transpose/layout 等派生字段不重置（SIMT 保留 Transpose 形态下
    // GetLoadTransposeSource 会在 CollectRewrittenBoundaries 兜底，不受影响）。
    if (pass > 0UL) {
      preparation.input.path.clear();
      preparation.index.path.clear();
      CollectInputPaths(indirect_load, template_id, preparation);
    }
    GE_ASSERT_SUCCESS(RewriteBroadcastPaths(indirect_load, template_id, preparation, analysis, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
    // RewriteInputPreNodes 末尾有 TopologicalSorting（幂等），多轮冗余但安全；
    // 排序只在最后一轮被 CollectRewrittenRegion 消费，如需优化可参数化控制时机。
    GE_ASSERT_SUCCESS(RewriteInputPreNodes(graph, indirect_load, template_id));
    // SIMD/SK 折叠一次性完成，无需多轮。
    if (template_id != ascir::TemplateId::kIndirectLoadSimt) {
      break;
    }
    if (ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex) ==
        producer_before) {
      break;
    }
    if (pass + 1UL == kMaxRewritePasses) {
      hit_cap = true;
    }
  }
  // 仅 SIMT 且真正跑满上限时 WARN（早退不刷；ValidateSimtTemplateRegion 的 INFO
  // 已覆盖通用拒绝，此处只补"循环尽了仍没收敛"的诊断信号）。
  if (template_id == ascir::TemplateId::kIndirectLoadSimt && hit_cap) {
    const auto final_boundary =
        ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
    if (final_boundary != nullptr && !af::ops::IsOps<af::ascir_op::Load>(final_boundary) &&
        GetLoadTransposeSource(final_boundary) == nullptr) {
      GELOGW("[IndirectLoad] SIMT boundary[%s(%s)] is not Load after rewrite loop cap.", final_boundary->GetNamePtr(),
             final_boundary->GetTypePtr());
    }
  }
  return af::SUCCESS;
}

af::Status AnalyzeRewrittenGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, bool &is_candidate_legal,
                                 RewrittenGraphAnalysis &analysis) {
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  // 分析阶段：路径、布局分类、对齐标志（只读）
  PhysicalViewPreparation preparation;
  GE_ASSERT_SUCCESS(PreparePhysicalViews(indirect_load, template_id, preparation, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(ValidateOutputStoreTargets(indirect_load, template_id, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  analysis.align_index_path = template_id != ascir::TemplateId::kIndirectLoadSimt &&
                              NeedsAlignedUbWindow(preparation.logical_view.index, axis_index);
  analysis.align_input_path = template_id != ascir::TemplateId::kIndirectLoadSimt &&
                              NeedsAlignedUbWindow(preparation.logical_view.input, axis_index);
  analysis.input_load_transposed = preparation.input.load_transpose != nullptr;
  analysis.index_load_transposed = preparation.index.load_transpose != nullptr;
  analysis.simd_index_uses_output_inner_axis = preparation.index.simd_index_uses_output_inner_axis;

  GE_ASSERT_SUCCESS(
      RewriteInputChainToFixpoint(graph, indirect_load, template_id, preparation, analysis, is_candidate_legal));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  // 收集阶段：一次遍历收集全部输出边界；SK 也必须知道 Norm 后置区域，
  // 但 SK 的输入/索引分区仍由其专用路径处理。SK 消费子图走通用调度，
  // 允许归约结果扇出到物化 Store 与 Elementwise 消费链的多分支形态。
  const bool allow_multi_consumer_reduce = template_id == ascir::TemplateId::kIndirectLoadSK;
  GE_ASSERT_SUCCESS(
      CollectRewrittenBoundaries(indirect_load, analysis, is_candidate_legal, allow_multi_consumer_reduce));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  if (template_id != ascir::TemplateId::kIndirectLoadSK) {
    GE_ASSERT_SUCCESS(CollectRewrittenRegion(graph, indirect_load, template_id, analysis));
  }

  // 路径布局处理：紧凑零 stride 写回物理视图，需对齐的 strided 路径标注 UB role。
  GE_ASSERT_SUCCESS(ApplyTemplatePathLayouts(indirect_load, template_id, preparation, analysis));
  return af::SUCCESS;
}

}  // namespace indirect_load_detail
}  // namespace optimize
