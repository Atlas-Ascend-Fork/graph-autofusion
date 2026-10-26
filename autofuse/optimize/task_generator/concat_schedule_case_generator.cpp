/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_schedule_case_generator.h"

#include <queue>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

#include "ascir/meta/ascir_utils.h"
#include "ascir/meta/ascir_ops_utils.h"
#include "optimize/schedule_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "optimize/task_generator/concat_group_partitioner.h"
#include "optimize/task_generator/concat_score_function_generator.h"
#include "optimize/task_generator/concat_inputs_unification_pass.h"
#include "platform/platform_factory.h"
#include "util/mem_utils.h"
#include "task_generator/cast_optimization_pass.h"
#include "common_utils.h"

namespace optimize {
namespace semantic = ascgen_utils::template_semantic;
namespace {
constexpr uint32_t kMaxInputNum = 48U;
constexpr size_t kTemplateSizeAll = 3UL;
constexpr int32_t kConcatAlgTranspose = 0;
constexpr int64_t kSmallDimSizeThreshold = 256;

void CollectInAndOutNodes(const af::NodePtr &node, std::set<af::Node *> &visited_nodes,
                          std::queue<af::NodePtr> &nodes) {
  for (const auto &in_node : node->GetInDataNodes()) {
    if (visited_nodes.emplace(in_node.get()).second) {
      nodes.emplace(in_node);
    }
  }
  for (const auto &out_node : node->GetOutDataNodes()) {
    if (visited_nodes.emplace(out_node.get()).second) {
      nodes.emplace(out_node);
    }
  }
}

void ReplaceAxisById(std::vector<ascir::AxisId> &axes, ascir::AxisId old_id, ascir::AxisId new_id) {
  for (auto &axis_id : axes) {
    if (axis_id == old_id) {
      axis_id = new_id;
    }
  }
}

bool NeedUpdate(const af::AscNodePtr &node, const size_t concat_dim) {
  if (ScheduleUtils::IsBuffer(node)) {
    return false;
  }
  const auto &output_repeat = node->outputs[0].attr.repeats;
  GE_WARN_ASSERT(concat_dim < output_repeat.size());
  return (!ascgen_utils::ExpressEq(output_repeat[concat_dim], af::ops::One));
}
}  // namespace

Status ConcatFusionCaseGenerator::AddTemplatesForFirstDimConcat(const af::AscNodePtr &concat_node,
                                                                ascir::HintGraph &graph,
                                                                std::vector<ascir::ImplGraph> &graphs) {
  const bool is_one_axis = (concat_node->outputs[0].attr.repeats.size() == 1UL);
  const bool is_single_input = concat_node->inputs.Size() == 1U;
  if ((!is_single_input) && (!convert_to_store_) && (!support_small_tail_) && (is_one_axis || (concat_dim_ > 0))) {
    // 单维concat, 在前面补轴，复用非首轴处理逻辑
    // 先限制单维，后续处理多维但小包场景
    if (is_one_axis) {
      GE_ASSERT_SUCCESS(InsertAxis(graph), "Failed to insert axis for graph:[%s].", graph.GetName().c_str());
      concat_dim_ = 1;
      GE_ASSERT_SUCCESS(AddTemplateIfCanFitInOneKernel(concat_node, graph, graphs));
    }
    GE_ASSERT_SUCCESS(AddTemplateForSplitConcat(graph, graphs));
    GE_ASSERT_SUCCESS(MarkNoMergeFirstAxis(graphs));
    return af::SUCCESS;
  }
  // gm concat
  GE_CHK_STATUS_RET(Prepare(concat_node, concat_dim_), "Prepare failed");
  GE_CHK_STATUS_RET(ConvertConcatToStores(graph, concat_node), "ConvertConcatToStores failed");
  graphs.emplace_back(graph);
  GELOGI("concat on first dim, num_inputs = %u, 1 template was generated", concat_node->inputs.Size());
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                           std::vector<std::string> &score_functions) {
  bool has_unsupported_op = false;
  const auto concat_nodes = FindConcatNodes(graph, &has_unsupported_op);
  if (concat_nodes.empty()) {
    return af::SUCCESS;
  }
  auto &concat_node = concat_nodes.front();
  bool is_first_dim = false;
  GE_CHK_STATUS_RET(ScheduleUtils::ResolveDiffDim(concat_node, concat_dim_, is_first_dim), "ResolveConcatDim failed");
  GE_ASSERT_SUCCESS(AddExtraShapeEnv(concat_node, concat_dim_));
  GE_ASSERT_SUCCESS(SplitDataForDifferentConcatDim(graph), "Failed to split data for graph:[%s].",
                    graph.GetName().c_str());
  if (has_unsupported_op) {
    GE_CHK_STATUS_RET(Prepare(concat_node, concat_dim_), "Prepare failed");
    GE_CHK_STATUS_RET(ConvertConcatToStores(graph, concat_node), "ConvertConcatToStores failed");
    return af::SUCCESS;
  }

  const auto backend_spec = BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  support_small_tail_ = backend_spec->concat_alg == kConcatAlgTranspose;
  // case1. 首轴concat，转store
  if (is_first_dim) {
    return AddTemplatesForFirstDimConcat(concat_node, graph, graphs);
  }

  // case2. 生成ub内concat的case
  if ((concat_node->inputs.Size() <= kMaxInputNum)) {
    graphs.emplace_back(graph);
    // 如果匹配小尾轴, UB能全载，则不需要生成case3模板
    if (support_small_tail_ && ascir::utils::UseSmallTailConcatApi(*concat_node)) {
      GELOGI("match small tail pattern, 1 template was generated");
      return af::SUCCESS;
    }
  }

  // case3. 在ub不全载的case的图中, 对concat进行分组
  GE_ASSERT_SUCCESS(AddTemplateForSplitConcat(graph, graphs));

  // 如果是动态shape, 添加small tail模板，运行时通过score func选择
  if (NeedDynSmallTailTemplate(concat_node)) {
    GE_ASSERT_SUCCESS(AddTemplateForSmallTail(graph, graphs));
  }

  GE_ASSERT_SUCCESS(RunCastOptimizationPass(graphs));
  if (!support_small_tail_) {
    GE_ASSERT_SUCCESS(ConcatInputUnificationPass::Run(graphs));
  }

  // 多模板, 为ub concat模板提供打分函数
  GE_ASSERT_SUCCESS(GenerateScoreFunctions(graphs, concat_dim_, score_functions));
  return af::SUCCESS;
}

ConcatFusionCaseGenerator &ConcatFusionCaseGenerator::SetConvertToStoreMode() {
  convert_to_store_ = true;
  return *this;
}

Status ConcatFusionCaseGenerator::AddTemplateForSplitConcat(const ascir::HintGraph &graph,
                                                            std::vector<ascir::ImplGraph> &graphs) {
  ascir::ImplGraph optimized_graph((graph.GetName() + "_group_concat").c_str());
  GE_ASSERT_TRUE(optimized_graph.CopyFrom(graph));
  auto concat_node = FindConcatNodes(optimized_graph).front();
  GE_CHK_STATUS_RET(Prepare(concat_node, concat_dim_), "Prepare failed");
  split_concat_ = false;
  GE_CHK_STATUS_RET(SplitConcats(optimized_graph, concat_node, split_concat_), "SplitConcats failed");
  GELOGI("Concat on non-first dim, split concat into groups templates generated, split = %d",
         static_cast<int32_t>(split_concat_));
  if (split_concat_) {
    if (concat_node->outputs[0].attr.repeats[concat_dim_].IsConstExpr() && (!KeepOriginGraph(concat_node))) {
      // 经过split的性能更好, 防止ATT选错, 删除非Split的
      graphs.clear();
    }
  } else {
    GE_CHK_STATUS_RET(ConvertConcatToStores(optimized_graph, concat_node), "ConvertConcatToStores failed");
  }
  graphs.emplace_back(optimized_graph);
  return af::SUCCESS;
}

bool ConcatFusionCaseGenerator::NeedDynSmallTailTemplate(const af::AscNodePtr &concat_node) const {
  const auto dtype_size = GetSizeByDataType(concat_node->outputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0);
  return support_small_tail_ && ((dtype_size == sizeof(uint16_t)) || (dtype_size == sizeof(uint32_t))) &&
         (concat_node->inputs.Size() <= kMaxInputNum) &&
         (!concat_node->outputs[0].attr.strides[concat_dim_ - 1].IsConstExpr());
}

Status ConcatFusionCaseGenerator::AddTemplateForSmallTail(const ascir::HintGraph &graph,
                                                          std::vector<ascir::ImplGraph> &graphs) {
  GELOGI("exists dynamic dim after concat_dim, generate force small tail template");
  ascir::ImplGraph force_small_tail_graph((graph.GetName() + "_force_small_tail").c_str());
  GE_ASSERT_TRUE(force_small_tail_graph.CopyFrom(graph));
  auto force_small_tail_node = FindConcatNodes(force_small_tail_graph).front();
  GE_ASSERT_TRUE(af::AttrUtils::SetBool(force_small_tail_node->GetOpDesc(), "_concat_small_tail", true));
  graphs.emplace_back(force_small_tail_graph);
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::GenerateScoreFunctions(const std::vector<ascir::ImplGraph> &graphs, size_t concat_dim,
                                                         std::vector<std::string> &score_functions) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((graphs.size() > 1U), af::SUCCESS);
  if (support_small_tail_) {
    if (!has_recompute_) {
      score_functions.resize(graphs.size());
      const auto concat_node = FindConcatNodes(graphs.front()).front();
      GE_CHK_STATUS_RET(
          ConcatScoreFunctionGenerator(graphs.front(), concat_node, concat_dim).Generate(score_functions.front()),
          "Failed to generate score func for ub_concat");
      if (graphs.size() == kTemplateSizeAll) {
        GE_CHK_STATUS_RET(ConcatScoreFunctionGenerator(graphs.back(), concat_node, concat_dim)
                              .GenerateForCheckSmallTail(score_functions.back()),
                          "Failed to generate score func for small_tail_concat");
      }
    }
  } else {
    // 没有分组时, 优先UB全载的模板
    if (!split_concat_) {
      const auto concat_node = FindConcatNodes(graphs.front()).front();
      constexpr uint32_t kMaxNumInputs = 16;
      if ((!has_recompute_) && (concat_node->inputs.Size() <= kMaxNumInputs) &&
          IsSmallBlock(concat_node, concat_dim_)) {
        score_functions.resize(graphs.size());
        ConcatScoreFunctionGenerator::GenerateScoreOne(score_functions.front());
      }
    }
  }
  return af::SUCCESS;
}

std::vector<af::AscNodePtr> ConcatFusionCaseGenerator::FindConcatNodes(const ascir::HintGraph &owner_graph,
                                                                       bool *has_unsupported_op) {
  if (has_unsupported_op != nullptr) {
    *has_unsupported_op = false;
  }
  std::vector<af::AscNodePtr> concat_nodes;
  for (const auto &node : owner_graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Concat>(node)) {
      concat_nodes.emplace_back(node);
    } else if (has_unsupported_op != nullptr && (!*has_unsupported_op)) {
      const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
      if ((asc_node != nullptr) && (ScheduleUtils::IsTranspose(asc_node) || ScheduleUtils::IsReduce(asc_node))) {
        *has_unsupported_op = true;
        GELOGI("graph contains Transpose/Reduce node %s", node->GetNamePtr());
      }
    } else {
      // do nothing
    }
  }
  return concat_nodes;
}

Status ConcatFusionCaseGenerator::ConvertSingleInput(ascir::HintGraph &owner_graph, const af::AscNodePtr &concat_node,
                                                     size_t in_index, size_t group_idx,
                                                     ConcatDimAxisMap &repeat_to_axis_id) {
  const auto &all_in_data_anchors = concat_node->GetAllInDataAnchors();
  const auto &loop_axis = owner_graph.GetAllAxis();

  const auto &concat_in_anchor = all_in_data_anchors.at(in_index);
  auto input_repeat = concat_node->inputs[in_index].attr.repeats[concat_dim_];

  if (repeat_to_axis_id.find(input_repeat) == repeat_to_axis_id.end()) {
    auto new_axis =
        owner_graph.CreateAxis(loop_axis[concat_dim_]->name + "_" + std::to_string(group_idx), input_repeat);
    repeat_to_axis_id[input_repeat] = new_axis.id;
    GELOGD("Create axis [%s, %ld] for input repeat=[%s].", new_axis.name.c_str(), new_axis.id,
           af::SymbolicUtils::ToString(input_repeat).c_str());
  }
  auto replace_axis = owner_graph.FindAxis(repeat_to_axis_id[input_repeat]);
  GE_ASSERT_NOTNULL(replace_axis);

  return ReplaceWithStore(concat_node, concat_in_anchor, *replace_axis);
}

Status ConcatFusionCaseGenerator::ConvertConcatToStores(ascir::HintGraph &owner_graph,
                                                        const af::AscNodePtr &concat_node) {
  ConcatGroupPartitioner partitioner(concat_node, concat_dim_);
  GE_ASSERT_SUCCESS(partitioner.RecomputeDiffAxes());
  has_recompute_ = partitioner.HasRecompute();
  GE_ASSERT_SUCCESS(PrepareForModifyingGraph(concat_node));
  ConcatDimAxisMap repeat_to_axis_id;
  const auto all_in_data_anchors_count = concat_node->GetAllInDataAnchors().size();  // 只需获取一次大小
  for (size_t i = 0UL; i < all_in_data_anchors_count; ++i) {
    const auto in_index = all_in_data_anchors_count - i - 1UL;
    GE_ASSERT_SUCCESS(ConvertSingleInput(owner_graph, concat_node, in_index, i, repeat_to_axis_id),
                      "ProcessSingleInput failed in ConvertConcatToStores");
  }

  GE_CHK_STATUS_RET(RemoveUnusedNodes(concat_node, post_concat_nodes_), "RemoveUnusedNodes failed");
  GE_CHK_STATUS_RET(PassUtils::PruneGraph(owner_graph), "Prune graph failed");
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(owner_graph));
  GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticNameForGraph(owner_graph, semantic::kConcatOffsetStore));
  ascir::utils::DumpGraph(owner_graph, "AfterConvertConcatToStore");
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::SplitConcats(ascir::HintGraph &owner_graph, const af::AscNodePtr &concat_node,
                                               bool &split) {
  std::vector<ConcatGroupPartitioner::ConcatGroup> groups;
  ConcatGroupPartitioner partitioner(concat_node, concat_dim_);
  GE_ASSERT_SUCCESS(partitioner.PartitionGroups(groups));
  if ((groups.size() <= 1U) || (groups.size() == concat_node->inputs.Size())) {
    return af::SUCCESS;
  }
  has_recompute_ = partitioner.HasRecompute();
  GE_ASSERT_SUCCESS(PrepareForModifyingGraph(concat_node));
  for (size_t i = 0U; i < groups.size(); ++i) {
    const auto &group = groups[i];
    GELOGI("group[%zu] start = %ld, end = %ld", i, group.start, group.end);
  }
  ConcatDimAxisMap repeat_to_axis_id;
  auto loop_axis = owner_graph.GetAllAxis();
  for (size_t i = 0U; i < groups.size(); ++i) {
    const auto &group = groups[groups.size() - i - 1];
    if (group.end - group.start == 1) {
      GE_ASSERT_SUCCESS(ConvertSingleInput(owner_graph, concat_node, group.start, i, repeat_to_axis_id),
                        "ProcessSingleInput failed in ConvertConcatToStores");
    } else {
      // loads -> concat -> store
      GE_CHK_STATUS_RET(ReplaceWithConcat(owner_graph, concat_node, group.start, group.end));
    }
  }
  GE_CHK_STATUS_RET(RemoveUnusedNodes(concat_node, post_concat_nodes_), "RemoveUnusedNodes failed");
  GE_CHK_STATUS_RET(PassUtils::PruneGraph(owner_graph), "Prune graph failed");
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(owner_graph));
  if (split) {
    GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticNameForGraph(owner_graph, semantic::kConcatOffsetStore));
  }
  ascir::utils::DumpGraph(owner_graph, "AfterSplitConcat");
  split = true;
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::Prepare(const af::AscNodePtr &concat_node, size_t concat_dim) {
  af::Expression dim_offset = af::ops::Zero;
  for (const auto &in_anchor : concat_node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    concat_dim_offsets_.emplace_back(dim_offset);
    dim_offset = dim_offset + concat_node->inputs[static_cast<uint32_t>(in_anchor->GetIdx())].attr.repeats[concat_dim];
  }
  concat_axis_id_ = concat_node->attr.sched.axis[concat_dim];
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::PropagateAxisChanges(af::Node *start_node, ascir::AxisId old_axis_id,
                                                       ascir::AxisId new_axis_id) {
  std::set<af::Node *> visited_nodes;
  std::queue<af::Node *> node_queue;

  visited_nodes.emplace(start_node);
  node_queue.emplace(start_node);

  while (!node_queue.empty()) {
    const auto curr_node = dynamic_cast<af::AscNode *>(node_queue.front());
    node_queue.pop();
    GE_ASSERT_NOTNULL(curr_node);
    if (curr_node->attr.api.type != af::ApiType::kAPITypeBuffer) {
      ReplaceAxisById(curr_node->attr.sched.axis, old_axis_id, new_axis_id);
      for (const auto &out_tensor : curr_node->outputs()) {
        ReplaceAxisById(out_tensor->attr.axis, old_axis_id, new_axis_id);
      }
    }

    for (const auto &out_node : curr_node->GetOutDataNodes()) {
      if (visited_nodes.count(out_node.get()) == 0UL) {
        visited_nodes.emplace(out_node.get());
        node_queue.emplace(out_node.get());
      }
    }

    for (const auto &in_node : curr_node->GetInDataNodes()) {
      if (visited_nodes.count(in_node.get()) == 0UL) {
        visited_nodes.emplace(in_node.get());
        node_queue.emplace(in_node.get());
      }
    }
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::ReplaceWithStore(const af::AscNodePtr &concat_node,
                                                   const af::InDataAnchorPtr &concat_in_anchor,
                                                   const af::Axis &replace_axis) {
  const auto in_index = concat_in_anchor->GetIdx();
  const auto src_out_anchor = concat_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_out_anchor);
  concat_in_anchor->UnlinkAll();
  auto src_node = dynamic_cast<af::AscNode *>(src_out_anchor->GetOwnerNodeBarePtr());
  GE_ASSERT_NOTNULL(src_node);
  const auto old_axis_id = concat_node->attr.sched.axis[concat_dim_];
  const auto new_axis_id = replace_axis.id;
  GE_CHK_STATUS_RET(PropagateAxisChanges(src_node, old_axis_id, new_axis_id),
                    "PropagateAxisChanges failed in ReplaceWithStore");
  std::vector<af::InDataAnchorPtr> dst_in_anchors;
  std::unordered_map<std::string, af::NodePtr> name_to_new_node;
  GE_ASSERT_SUCCESS(CloneNonConcatNodes(replace_axis, old_axis_id, in_index, dst_in_anchors, name_to_new_node));
  for (const auto &peer_in_anchor : dst_in_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(src_out_anchor, peer_in_anchor));
    GE_ASSERT_SUCCESS(ReconnectIfShareSameAncestor(name_to_new_node, peer_in_anchor));
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::ReplaceWithConcat(ascir::ImplGraph &owner_graph, const af::AscNodePtr &concat_node,
                                                    size_t start, size_t end) {
  auto suffix = "_" + std::to_string(start) + "_" + std::to_string(end);
  af::ascir_op::Concat concat_op((concat_node->GetName() + suffix).c_str());
  SetConcatOpAttr(concat_op, concat_node, concat_dim_, start, end);
  auto new_concat_node = owner_graph.AddNode(concat_op);
  GE_ASSERT_NOTNULL(new_concat_node);
  GELOGD("split concat [%zu, %zu), output repeats = %s", start, end,
         af::ToString(new_concat_node->outputs[0].attr.repeats).c_str());
  for (size_t i = end; i > start; --i) {
    auto concat_in_anchor = concat_node->GetInDataAnchor(static_cast<int32_t>(i) - 1);
    GE_CHECK_NOTNULL(concat_in_anchor);
    auto peer_out_anchor = concat_in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    GE_CHK_STATUS_RET(af::GraphUtils::RemoveEdge(peer_out_anchor, concat_in_anchor), "Failed to RemoveEdge");
    GE_CHK_STATUS_RET(af::GraphUtils::AddEdge(peer_out_anchor, new_concat_node->GetInDataAnchor(i - start - 1)),
                      "Failed to AddEdge");
  }
  const auto &output_repeats = new_concat_node->outputs[0].attr.repeats;
  ascir::Axis concat_axis = *(owner_graph.GetAllAxis().at(concat_axis_id_));
  auto new_concat_axis =
      owner_graph.CreateAxis(concat_axis.name + "_ss_" + std::to_string(start), output_repeats[concat_dim_]);

  // 前向刷轴
  const auto old_axis_id = concat_node->attr.sched.axis[concat_dim_];
  const auto new_axis_id = new_concat_axis.id;
  GE_CHK_STATUS_RET(PropagateAxisChanges(new_concat_node.get(), old_axis_id, new_axis_id),
                    "PropagateAxisChanges failed in ReplaceWithConcat");
  GELOGD("New axis %s, size = %s", new_concat_axis.name.c_str(),
         af::SymbolicUtils::ToString(new_concat_axis.size).c_str());
  std::vector<af::InDataAnchorPtr> dst_in_anchors;
  GE_ASSERT_SUCCESS(ReplaceAxis(new_concat_node, old_axis_id, new_concat_axis, true));
  std::unordered_map<std::string, af::NodePtr> name_to_new_node;
  GE_ASSERT_SUCCESS(CloneNonConcatNodes(new_concat_axis, old_axis_id, start, dst_in_anchors, name_to_new_node));
  for (const auto &in_anchor : new_concat_node->GetAllInDataAnchors()) {
    GE_ASSERT_SUCCESS(ReconnectIfShareSameAncestor(name_to_new_node, in_anchor));
  }
  for (const auto &peer_in_anchor : dst_in_anchors) {
    GE_ASSERT_GRAPH_SUCCESS(af::GraphUtils::AddEdge(new_concat_node->GetOutDataAnchor(0), peer_in_anchor));
  }
  return af::SUCCESS;
}

af::Status ConcatFusionCaseGenerator::SetConcatOpAttr(af::ascir_op::Concat &concat_op,
                                                      const af::AscNodePtr &concat_node, size_t concat_dim,
                                                      size_t start, size_t end) {
  GE_ASSERT_TRUE(end <= concat_node->inputs.Size());
  auto repeats = concat_node->inputs[start].attr.repeats;
  for (size_t i = start + 1U; i < end; ++i) {
    repeats[concat_dim] = repeats[concat_dim] + concat_node->inputs[i].attr.repeats[concat_dim];
  }
  af::Expression stride = af::sym::kSymbolOne;
  std::vector<af::Expression> strides(repeats.size(), af::sym::kSymbolOne);
  for (auto i = static_cast<int32_t>(repeats.size() - 1); i >= 0; --i) {
    strides[i] = stride;
    stride = (stride * repeats[i]);
  }
  const auto &concat_output_tensor_attr = concat_node->outputs[0].attr;
  concat_op.attr = concat_node->attr;
  concat_op.y.dtype = concat_output_tensor_attr.dtype;
  *concat_op.y.repeats = repeats;
  *concat_op.y.strides = strides;
  *concat_op.y.axis = concat_output_tensor_attr.axis;
  concat_op.DynamicInputRegister("x", end - start);
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::RemoveUnusedNodes(const af::AscNodePtr &concat_node,
                                                    const std::vector<af::AscNodePtr> &nodes) {
  auto owner_compute_graph = concat_node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_compute_graph);
  GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(concat_node), "Failed to remove node: %s",
                    concat_node->GetNamePtr());
  for (const auto &node : nodes) {
    GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(node), "Failed to remove node: %s", node->GetNamePtr());
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::SplitDataForDifferentConcatDim(ascir::ImplGraph &owner_graph) {
  for (const auto &node : owner_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!ScheduleUtils::IsDataInput(node)) {
      continue;
    }
    auto output_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(output_anchor);
    auto peer_in_anchors = output_anchor->GetPeerInDataAnchors();
    for (size_t idx = 1UL; idx < peer_in_anchors.size(); ++idx) {
      std::string node_name = node->GetName() + std::string("_") + std::to_string(idx);
      af::AscNodePtr data_node;
      if (af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
        af::ascir_op::ScalarData scalar_data(node_name.c_str());
        data_node = owner_graph.AddNode(scalar_data);
      } else {
        af::ascir_op::Data data(node_name.c_str());
        data_node = owner_graph.AddNode(data);
      }
      GE_ASSERT_NOTNULL(data_node);
      data_node->attr = node->attr;
      data_node->outputs[0].attr = node->outputs[0].attr;
      auto new_out_anchor = data_node->GetOutDataAnchor(0);
      GE_ASSERT_NOTNULL(new_out_anchor);
      GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(output_anchor, peer_in_anchors.at(idx)));
      GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(new_out_anchor, peer_in_anchors.at(idx)));
    }
  }
  return af::SUCCESS;
}

af::Status ConcatFusionCaseGenerator::CollectBackwardNodes(const af::NodePtr &concat_node, bool collect_all,
                                                           std::vector<af::AscNodePtr> &nodes) const {
  std::set<af::Node *> visited_nodes{concat_node.get()};
  std::queue<af::NodePtr> next_nodes;
  for (const auto &out_data_node : concat_node->GetOutDataNodes()) {
    if (visited_nodes.emplace(out_data_node.get()).second) {
      next_nodes.push(out_data_node);
    }
  }
  while (!next_nodes.empty()) {
    auto &top = next_nodes.front();
    auto asc_node = std::dynamic_pointer_cast<af::AscNode>(top);
    GE_ASSERT_NOTNULL(asc_node);
    if (collect_all || NeedUpdate(asc_node, concat_dim_)) {
      nodes.emplace_back(asc_node);
      CollectInAndOutNodes(top, visited_nodes, next_nodes);
    } else {
      GELOGD("skip node: %s", asc_node->GetNamePtr());
    }
    next_nodes.pop();
  }
  std::sort(nodes.begin(), nodes.end(), [](const af::AscNodePtr &lhs, const af::AscNodePtr &rhs) -> bool {
    return lhs->GetOpDesc()->GetId() < rhs->GetOpDesc()->GetId();
  });
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::CloneNonConcatNodes(const af::Axis &new_axis, ascir::AxisId old_axis_id, size_t index,
                                                      std::vector<af::InDataAnchorPtr> &in_anchors,
                                                      std::unordered_map<std::string, af::NodePtr> &name_to_new_node) {
  GE_ASSERT_TRUE(!post_concat_nodes_.empty());
  ascir::ImplGraph owner_graph("owner_graph");
  GE_ASSERT_SUCCESS(
      af::AscGraphUtils::FromComputeGraph(post_concat_nodes_.front()->GetOwnerComputeGraph(), owner_graph));
  std::string suffix;
  if (index != 0UL) {
    suffix = "_split_" + std::to_string(index);
  }
  std::unordered_map<std::string, af::NodePtr> all_new_nodes;
  for (const auto &asc_node : post_concat_nodes_) {
    const auto &op_desc = af::GraphUtils::CopyOpDesc(asc_node->GetOpDesc(), nullptr);
    GE_CHECK_NOTNULL(op_desc);
    op_desc->SetName(asc_node->GetName() + suffix);
    af::Operator op = af::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = owner_graph.AddNode(op);
    all_new_nodes[dst_new_node->GetName()] = dst_new_node;
    name_to_new_node[asc_node->GetName()] = dst_new_node;
    GE_ASSERT_TRUE(af::AscGraph::CopyAscNodeTensorAttr(asc_node, dst_new_node),
                   "DoCopyAscNodeTensorAttr failed, node = %s[%s]", asc_node->GetNamePtr(), asc_node->GetTypePtr());
    const auto need_update = nodes_need_update_axis_size_.find(asc_node.get()) != nodes_need_update_axis_size_.cend();
    GELOGD("Node: %s need_update = %d", asc_node->GetNamePtr(), static_cast<int32_t>(need_update));
    if (need_update) {
      GE_ASSERT_SUCCESS(UpdateOffsetAttr(dst_new_node, index));
    }
    if (const auto it = out_node_name_to_indices_.find(asc_node->GetName()); it != out_node_name_to_indices_.cend()) {
      for (const auto in_anchor_index : it->second) {
        in_anchors.emplace_back(dst_new_node->GetInDataAnchor(in_anchor_index));
      }
    }
    if (!ScheduleUtils::IsBuffer(dst_new_node)) {
      GE_ASSERT_SUCCESS(ReplaceAxis(dst_new_node, old_axis_id, new_axis, need_update));
    }
  }
  for (const auto &src_node : post_concat_nodes_) {
    GE_CHK_STATUS_RET(af::GraphUtils::RelinkGraphEdges(src_node, suffix, all_new_nodes), "RelinkGraphEdges failed");
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::UpdateOffsetAttr(const af::AscNodePtr &dst_new_node, size_t index) {
  if (dst_new_node->GetType() == af::ascir_op::Store::Type) {
    const auto offset = concat_dim_offsets_[index] * dst_new_node->outputs[0].attr.strides[concat_dim_];
    const auto ir_attr = dst_new_node->attr.ir_attr->DownCastTo<af::ascir_op::Store::AscStoreIrAttrDef>();
    GE_ASSERT_NOTNULL(ir_attr);
    GE_CHK_STATUS_RET(ir_attr->SetOffset(offset), "Failed to set offset to %s", dst_new_node->GetNamePtr());
    GELOGI("Store node: %s added, offset = %s", dst_new_node->GetName().c_str(), offset.Serialize().get());
  } else if (dst_new_node->GetType() == af::ascir_op::Load::Type) {
    const auto offset = concat_dim_offsets_[index] * dst_new_node->outputs[0].attr.strides[concat_dim_];
    const auto ir_attr = dst_new_node->attr.ir_attr->DownCastTo<af::ascir_op::Load::AscLoadIrAttrDef>();
    GE_ASSERT_NOTNULL(ir_attr);
    GE_CHK_STATUS_RET(ir_attr->SetOffset(offset), "Failed to set offset to %s", dst_new_node->GetNamePtr());
    GELOGI("Load node: %s added, offset = %s", dst_new_node->GetName().c_str(), offset.Serialize().get());
  } else {
    // do nothing
  }
  return af::SUCCESS;
}

af::Status ConcatFusionCaseGenerator::ReplaceAxis(const af::AscNodePtr &node, ascir::AxisId old_axis_id,
                                                  const af::Axis &to_axis, bool update_axis_size) {
  for (int64_t &axis_id : node->attr.sched.axis) {
    if (axis_id == old_axis_id) {
      axis_id = to_axis.id;
    }
  }

  for (uint32_t i = 0U; i < node->outputs().size(); ++i) {
    GE_ASSERT_SUCCESS(UpdateOutputAttr(node, old_axis_id, to_axis, node->outputs[i].attr, update_axis_size),
                      "Failed to update repeat and strides for outputs[%u], node = %s(%s)", i, node->GetNamePtr(),
                      node->GetTypePtr());
  }
  GELOGD("Replace axis for node: %s(%s) success", node->GetNamePtr(), node->GetTypePtr());
  return af::SUCCESS;
}

af::Status ConcatFusionCaseGenerator::UpdateOutputAttr(const af::AscNodePtr &node, ascir::AxisId old_axis_id,
                                                       const af::Axis &to_axis, af::AscTensorAttr &tensor_attr,
                                                       bool update_axis_size) {
  size_t axis_index = std::numeric_limits<size_t>::max();
  for (size_t i = 0UL; i < tensor_attr.axis.size(); ++i) {
    if (tensor_attr.axis[i] == old_axis_id) {
      tensor_attr.axis[i] = to_axis.id;
      axis_index = i;
    } else if (tensor_attr.axis[i] == to_axis.id) {
      axis_index = i;
    }
  }
  GE_ASSERT_TRUE(axis_index != std::numeric_limits<size_t>::max(),
                 "axis id %ld (old) / %ld (new) not found in node %s(%s) tensor_attr.axis", old_axis_id, to_axis.id,
                 node->GetNamePtr(), node->GetTypePtr());

  auto &repeats = tensor_attr.repeats;
  auto &strides = tensor_attr.strides;
  GELOGD("before update, repeats = %s, strides = %s", af::ToString(repeats).c_str(), af::ToString(strides).c_str());
  GE_ASSERT_TRUE(repeats.size() == strides.size());
  GE_ASSERT_TRUE(axis_index < repeats.size(), "axis_index = %zu, out of range [0, %zu)", axis_index, repeats.size());
  // concat_dim在brc轴, repeats和strides都不需要update
  if ((!update_axis_size) ||
      af::SymbolicUtils::StaticCheckEq(repeats[axis_index], af::ops::One) == af::TriBool::kTrue) {
    return af::SUCCESS;
  }
  repeats[axis_index] = to_axis.size;
  // Load/Store的stride为GM的stride, 不需要改变
  if (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node)) {
    if (af::SymbolicUtils::StaticCheckEq(to_axis.size, af::ops::One) == af::TriBool::kTrue) {
      node->outputs[0].attr.strides[axis_index] = af::ops::Zero;
    }
    return af::SUCCESS;
  }
  af::Expression stride = af::ops::One;
  for (auto i = static_cast<int32_t>(repeats.size() - 1); i >= 0; --i) {
    if (i != static_cast<int32_t>(repeats.size() - 1UL)) {
      stride = stride * repeats[i + 1];
    }
    if (af::SymbolicUtils::StaticCheckEq(repeats[i], af::ops::One) == af::TriBool::kTrue) {
      strides[i] = af::ops::Zero;
    } else {
      strides[i] = stride;
    }
  }
  GELOGD("after update, repeats = %s, strides = %s", af::ToString(repeats).c_str(), af::ToString(strides).c_str());
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::InsertAxis(const ascir::ImplGraph &optimized_graph) {
  const auto graph_attr =
      af::AscGraphUtils::GetComputeGraph(optimized_graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  GELOGD("before: axes = %s", ScheduleUtils::AxesToString(graph_attr->axis).c_str());
  const auto src_axes = graph_attr->axis;  // copy
  std::vector<af::AxisPtr> new_axes;
  for (const auto &src_axis : src_axes) {
    std::shared_ptr<af::Axis> new_axis = af::MakeShared<af::Axis>();
    GE_CHECK_NOTNULL(new_axis, "create axis failed");
    new_axis->id = src_axis->id;
    new_axis->name = src_axis->name;
    new_axis->type = src_axis->type;
    new_axis->size = src_axis->size;
    new_axes.push_back(std::move(new_axis));
  }

  auto new_axis_id = static_cast<int64_t>(new_axes.size());
  std::shared_ptr<af::Axis> const_axis = af::MakeShared<af::Axis>();
  GE_CHECK_NOTNULL(const_axis, "create axis failed");
  const_axis->id = static_cast<int64_t>(new_axes.size());
  const_axis->name = "axis_1d";
  const_axis->type = af::Axis::kAxisTypeOriginal;
  const_axis->size = af::ops::One;
  new_axes.push_back(std::move(const_axis));
  graph_attr->axis = std::move(new_axes);
  GELOGD("after: axes = %s", ScheduleUtils::AxesToString(graph_attr->axis).c_str());

  for (const auto &node : optimized_graph.GetAllNodes()) {
    if (ScheduleUtils::IsIOBuffer(node)) {
      continue;
    }
    auto cur_axis_ids = node->attr.sched.axis;
    node->attr.sched.axis.insert(node->attr.sched.axis.begin(), new_axis_id);
    for (const auto output_attr : node->outputs()) {
      output_attr->attr.axis.insert(output_attr->attr.axis.begin(), new_axis_id);
      if (output_attr->attr.strides[0UL] == 0) {
        output_attr->attr.strides.insert(output_attr->attr.strides.begin(), af::ops::One);
      } else {
        output_attr->attr.strides.insert(output_attr->attr.strides.begin(),
                                         af::sym::Mul(output_attr->attr.repeats[0UL], output_attr->attr.strides[0UL]));
      }
      output_attr->attr.repeats.insert(output_attr->attr.repeats.begin(), af::ops::One);
    }
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::AddTemplateIfCanFitInOneKernel(const af::AscNodePtr &concat_node,
                                                                 ascir::HintGraph &graph,
                                                                 std::vector<ascir::ImplGraph> &graphs) {
  GE_CHK_BOOL_RET_SPECIAL_STATUS(concat_node->inputs.Size() > kMaxInputNum, af::SUCCESS,
                                 "input num(%u) > max_input_num(%u), cannot fit in one kernel",
                                 concat_node->inputs.Size(), kMaxInputNum);
  const auto data_type_size = af::GetSizeByDataType(concat_node->outputs[0].attr.dtype);
  GE_ASSERT_TRUE(data_type_size > 0);
  const auto max_dim_size = kSmallDimSizeThreshold / data_type_size;
  for (uint32_t i = 0U; i < concat_node->inputs.Size(); ++i) {
    const auto dim_expr = concat_node->inputs[i].attr.repeats.back();
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!dim_expr.IsConstExpr()), af::SUCCESS, "input[%zu] is not known shape: %s", i,
                                   af::ToString(concat_node->inputs[i].attr.repeats).c_str());
    int64_t dim_size = -1;
    GE_ASSERT_TRUE(dim_expr.GetConstValue(dim_size));
    GE_CHK_BOOL_RET_SPECIAL_STATUS((dim_size > max_dim_size), af::SUCCESS,
                                   "input[%zu] dim_size(%ld) is larger than threshold(%ld)", i, dim_size, max_dim_size);
  }
  graphs.emplace_back(graph);
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::MarkNoMergeFirstAxis(const std::vector<ascir::ImplGraph> &graphs) {
  for (const auto &graph : graphs) {
    for (const auto &node : graph.GetAllNodes()) {
      if (af::ops::IsOps<af::ascir_op::Concat>(node)) {
        GE_ASSERT_TRUE(af::AttrUtils::SetBool(node->GetOpDesc(), "_keep_first_axis", true));
      }
    }
  }
  return af::SUCCESS;
}

bool ConcatFusionCaseGenerator::KeepOriginGraph(const af::AscNodePtr &concat_node) const {
  bool keep_origin = false;
  if (!support_small_tail_) {
    constexpr uint32_t kMaxNumInputs = 16;
    keep_origin =
        has_recompute_ || (concat_node->inputs.Size() <= kMaxNumInputs) || IsSmallBlock(concat_node, concat_dim_);
  }
  return keep_origin;
}

bool ConcatFusionCaseGenerator::IsSmallBlock(const af::AscNodePtr &concat_node, size_t concat_dim) {
  constexpr int64_t kMaxOutputBlockSize = 16 * 1024;
  const auto dtype_size = GetSizeByDataType(concat_node->outputs[0].attr.dtype);
  GE_WARN_ASSERT(dtype_size > 0);
  const auto &output_repeats = concat_node->outputs[0].attr.repeats;
  int64_t output_size = dtype_size;
  for (size_t i = concat_dim; i < output_repeats.size(); ++i) {
    auto &dim_expr = output_repeats[i];
    if (!dim_expr.IsConstExpr()) {
      output_size = -1;
      break;
    }
    int64_t dim_size = -1;
    GE_WARN_ASSERT(dim_expr.GetConstValue(dim_size));
    output_size *= dim_size;
  }
  const bool is_small_block = ((output_size >= 0) && (output_size < kMaxOutputBlockSize));
  GELOGI("output shape = %s, concat_dim = %zu, is_small_block = %d", af::ToString(output_repeats).c_str(), concat_dim,
         static_cast<int32_t>(is_small_block));
  return is_small_block;
}

Status ConcatFusionCaseGenerator::ReconnectIfShareSameAncestor(
    const std::unordered_map<std::string, af::NodePtr> &name_to_node, const af::InDataAnchorPtr &in_anchor) {
  auto src_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_anchor);
  auto src_node = src_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(src_node);
  const auto &it = name_to_node.find(src_node->GetName());
  if (it != name_to_node.end()) {
    in_anchor->UnlinkAll();
    GE_ASSERT_GRAPH_SUCCESS(in_anchor->LinkFrom(it->second->GetOutDataAnchor(src_anchor->GetIdx())));
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::AddExtraShapeEnv(const af::AscNodePtr &concat_node, size_t concat_dim) {
  const auto &output_repeats = concat_node->outputs[0].attr.repeats;
  // axis开始, concat各输入的轴大小一致
  // 符号库无法推导存在限制, 这里将多个轴进行组合进行guard
  for (uint32_t k = 1; k < concat_node->inputs.Size(); ++k) {
    const auto &input_repeats = concat_node->inputs[k].attr.repeats;
    auto input_axis_size = af::ops::One;
    auto output_axis_size = af::ops::One;
    for (size_t i = output_repeats.size() - 1; i > concat_dim; --i) {
      input_axis_size = input_axis_size * input_repeats[i];
      output_axis_size = output_axis_size * output_repeats[i];
      GE_LOGW_IF(!EXPECT_SYMBOL_EQ(input_axis_size, output_axis_size),
                 "expect axis eq failed, concat_dim = %zu, cur_dim = %zu, input_axis_size = %s, output_axis_size = %s",
                 concat_dim, i, input_axis_size.Str().get(), output_axis_size.Str().get());
    }
  }
  for (uint32_t k = 1; k < concat_node->inputs.Size(); ++k) {
    const auto &input_repeats = concat_node->inputs[k].attr.repeats;
    auto input_axis_size = af::ops::One;
    af::Expression output_axis_size = af::ops::One;
    for (size_t i = concat_dim + 1; i < output_repeats.size(); ++i) {
      input_axis_size = input_axis_size * input_repeats[i];
      output_axis_size = output_axis_size * output_repeats[i];
      GE_LOGW_IF(!EXPECT_SYMBOL_EQ(input_axis_size, output_axis_size),
                 "expect axis eq failed, concat_dim = %zu, cur_dim = %zu, input_axis_size = %s, output_axis_size = %s",
                 concat_dim, i, input_axis_size.Str().get(), output_axis_size.Str().get());
    }
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::PrepareForModifyingGraph(const af::AscNodePtr &concat_node) {
  GE_ASSERT_SUCCESS(CollectBackwardNodes(concat_node, true, post_concat_nodes_));
  std::vector<af::AscNodePtr> concat_axis_related_nodes;
  GE_ASSERT_SUCCESS(CollectBackwardNodes(concat_node, false, concat_axis_related_nodes));
  for (const auto &node : concat_axis_related_nodes) {
    nodes_need_update_axis_size_.insert(node.get());
  }
  for (const auto &in_anchor_and_node : af::NodeUtils::GetOutDataNodesWithAnchorByIndex(*concat_node, 0)) {
    out_node_name_to_indices_[in_anchor_and_node.second->GetName()].emplace_back(in_anchor_and_node.first->GetIdx());
  }
  return af::SUCCESS;
}

Status ConcatFusionCaseGenerator::RunCastOptimizationPass(std::vector<ascir::ImplGraph> &graphs) {
  const auto backend_spec = BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  for (auto &graph : graphs) {
    GE_ASSERT_SUCCESS(af::optimize::CastOptimizationPass::Run(graph, backend_spec->concat_alg));
  }
  return af::SUCCESS;
}
}  // namespace optimize
