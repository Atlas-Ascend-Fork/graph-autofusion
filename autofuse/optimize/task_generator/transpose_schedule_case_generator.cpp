/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_schedule_case_generator.h"
#include <queue>
#include <unordered_map>
#include "graph/utils/graph_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "ascir_utils.h"
#include "ascir_ops_utils.h"
#include "ascir_ops.h"
#include "indirect_load_utils.h"
#include "schedule_utils.h"
#include "graph_properties_cache.h"
#include "common_utils.h"

namespace optimize {
namespace semantic = ascgen_utils::template_semantic;
namespace {
constexpr int32_t kTransposeNoNeedUBConvertSize = 512;  // 以512Byte作为是否需要消除Transpose的阈值
constexpr char kTransposePrefix[] = "transpose_";
}  // namespace

std::vector<af::AscNodePtr> TransposeFusionCaseGenerator::FindTransposeNodes(const ascir::HintGraph &owner_graph) {
  std::vector<af::AscNodePtr> transpose_nodes;
  for (const auto &node : owner_graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      transpose_nodes.emplace_back(node);
    }
  }
  return transpose_nodes;
}

Status TransposeFusionCaseGenerator::TransposeNodeInputsAndOutputsCheck(const af::AscNodePtr &transpose_node) const {
  const auto &transpose_in_data_nodes = transpose_node->GetInDataNodes();
  GE_ASSERT_TRUE(transpose_in_data_nodes.size() == 1UL, "%zu nodes links to transpose node:%s",
                 transpose_in_data_nodes.size(), transpose_node->GetNamePtr());
  return af::SUCCESS;
}

void TransposeFusionCaseGenerator::UpdateAxisByPath(::ascir::ImplGraph &owner_graph, const af::NodePtr &input_node,
                                                    std::set<af::Node *> &visited_nodes,
                                                    const std::vector<int64_t> &reordered_axis,
                                                    const std::vector<int64_t> &reordered_sched_axis) const {
  std::queue<af::NodePtr> path_nodes;
  path_nodes.emplace(input_node);
  while (!path_nodes.empty()) {
    auto top = path_nodes.front();
    path_nodes.pop();
    visited_nodes.emplace(top.get());
    ::ascir::NodeView input_view = std::dynamic_pointer_cast<af::AscNode>(top);
    if (!ScheduleUtils::IsBuffer(input_view)) {
      owner_graph.ApplyTensorAxisReorder(input_view, reordered_axis);
      owner_graph.ApplySchedAxisReorder(input_view, reordered_sched_axis);
      if (af::ops::IsOps<af::ascir_op::Load>(input_view)) {
        const std::string old_name = input_view->GetName();
        if (old_name.rfind(kTransposePrefix, 0U) != 0U) {
          input_view->GetOpDesc()->SetName(kTransposePrefix + old_name);
        }
      }
    }
    // 向上遍历。 暂时不考虑transpose之上的节点有额外向下的分支，如果有额外分支，则不能将transpose消除。
    for (size_t idx = 0UL; idx < top->GetInDataNodesSize(); ++idx) {
      if (visited_nodes.count(top->GetInDataNodes().at(idx).get()) == 0UL) {
        path_nodes.push(top->GetInDataNodes().at(idx));
      }
    }
  }
}

void TransposeFusionCaseGenerator::UpdateAxis(ascir::HintGraph &graph, const af::AscNodePtr &transpose_node) const {
  std::vector<std::pair<int64_t, int64_t>> transpose_info;
  const auto &reordered_axis = transpose_node->outputs[0].attr.axis;
  const auto &reordered_sched_axis = transpose_node->attr.sched.axis;
  std::set<af::Node *> unused_visited_nodes;
  UpdateAxisByPath(graph, transpose_node, unused_visited_nodes, reordered_axis, reordered_sched_axis);
}

Status TransposeFusionCaseGenerator::TransposeConvertProcess(ascir::HintGraph &graph,
                                                             const af::AscNodePtr &transpose_node) const {
  GE_CHK_STATUS_RET(TransposeNodeInputsAndOutputsCheck(transpose_node), "TransposeNode Check failed");
  UpdateAxis(graph, transpose_node);

  auto owner_compute_graph = transpose_node->GetOwnerComputeGraph();
  GE_CHK_STATUS_RET(owner_compute_graph->RemoveNode(transpose_node), "Failed to remove node: %s",
                    transpose_node->GetNamePtr());

  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticNameForGraph(graph, semantic::kTransposeFold));
  ascir::utils::DumpGraph(graph, "AfterConvertTranspose");
  return af::SUCCESS;
}

Status TransposeFusionCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                              std::vector<std::string> &score_functions) {
  const auto indirect_load = ascgen_utils::indirect_load::FindIndirectLoadNode(graph);
  if (indirect_load != nullptr) {
    GELOGI("Graph %s has indirect load, skip transpose task generation; handled by IndirectLoad tasks.",
           graph.GetName().c_str());
    return af::SUCCESS;
  }
  /*
  单个Transpose场景：
    场景1： 尾轴转置， 需要UB重排，Transpose节点保留；
    场景2：非尾轴转置，尾轴大于等于512Byte，不需要UB重排，Transpose删除，刷新load/store表示；
    场景3：非尾轴转置，尾轴小于等于512Byte，需要UB重排，Transpose节点保留；
    同时提供两套模板，后续根据打分机制来选取
  多个Transpose场景：
    不生成保留模板，直接将所有Transpose行为推到load上
  */

  const auto transpose_nodes = FindTransposeNodes(graph);
  if (transpose_nodes.empty()) {
    GELOGI("No transpose node found, skip");
    return af::SUCCESS;
  }

  GraphPropertiesCache cache(graph);
  if (cache.HasReduce()) {
    // 存在不支持与Transpose融合的计算节点，直接在原图上消除所有Transpose
    auto remaining = FindTransposeNodes(graph);
    while (!remaining.empty()) {
      GE_CHK_STATUS_RET(TransposeConvertProcess(graph, remaining.front()), "TransposeConvertProcess failed");
      remaining = FindTransposeNodes(graph);
    }
    return af::SUCCESS;
  }

  if (transpose_nodes.size() == 1UL) {
    // 单个Transpose：生成保留和消除两套模板，通过打分选择
    graphs.emplace_back(graph);

    ascir::ImplGraph optimized_graph((graph.GetName() + "_group_transpose").c_str());
    optimized_graph.CopyFrom(graph);

    auto transpose_node = FindTransposeNodes(optimized_graph).front();
    GE_CHK_STATUS_RET(TransposeConvertProcess(optimized_graph, transpose_node), "TransposeConvertProcess failed");
    graphs.emplace_back(optimized_graph);

    transpose_node = FindTransposeNodes(graphs[0]).front();
    score_functions.resize(2U);
    GE_CHK_STATUS_RET(GenerateScoreFuncForUbReorder(graph, transpose_node, score_functions[0]),
                      "Failed to generate score func");
  } else {
    // 多个Transpose：仅生成消除模板，将所有Transpose行为推到load上
    ascir::ImplGraph optimized_graph((graph.GetName() + "_group_transpose").c_str());
    optimized_graph.CopyFrom(graph);

    auto remaining = FindTransposeNodes(optimized_graph);
    while (!remaining.empty()) {
      GE_CHK_STATUS_RET(TransposeConvertProcess(optimized_graph, remaining.front()), "TransposeConvertProcess failed");
      remaining = FindTransposeNodes(optimized_graph);
    }
    graphs.emplace_back(optimized_graph);
  }

  return af::SUCCESS;
}

Status TransposeFusionCaseGenerator::GenerateScoreFuncForUbReorder(const ascir::HintGraph &graph,
                                                                   const af::AscNodePtr &transpose_node,
                                                                   std::string &score_func) {
  return TransposeScoreFunctionGenerator(graph, transpose_node).Generate(score_func);
}

TransposeScoreFunctionGenerator::TransposeScoreFunctionGenerator(const ascir::HintGraph &graph,
                                                                 af::AscNodePtr transpose_node)
    : graph_(&graph), transpose_node_(std::move(transpose_node)) {}

Status TransposeScoreFunctionGenerator::ParseRepeat() {
  const auto last_idx = transpose_node_->inputs[0].attr.axis.size() - 1;
  repeat_ = transpose_node_->inputs[0].attr.repeats[last_idx];
  return af::SUCCESS;
}

Status TransposeScoreFunctionGenerator::Generate(std::string &score_func) {
  GE_CHK_STATUS_RET(ParseRepeat());
  ss_ << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  int32_t score = 0;

  GE_CHK_STATUS_RET(GetScoreByExpr(score));
  GenerateReturnValue(score);
  score_func = ss_.str();
  return af::SUCCESS;
}

void TransposeScoreFunctionGenerator::GenerateReturnValue(const int32_t score) {
  ss_ << "  return " << score << ";" << std::endl;
  ss_ << "}" << std::endl;
}

Status TransposeScoreFunctionGenerator::GetScoreByExpr(int32_t &score) const {
  // 使用表达式值获取分数值
  const auto last_idx = transpose_node_->inputs[0].attr.axis.size() - 1;
  const auto &input_tail_axis = transpose_node_->inputs[0].attr.axis[last_idx];
  const auto &output_tail_axis = transpose_node_->outputs[0].attr.axis[last_idx];

  if (input_tail_axis != output_tail_axis) {
    score = 1;
    return af::SUCCESS;
  }
  // 非尾轴转置需要根据尾轴大小确定分数
  int32_t dim = -1;
  if (!repeat_.GetHint(dim)) {
    GELOGI("The repeat of axis %d is not constant, skip score calc.", dim);
    score = 0;
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(repeat_.GetHint(dim), "Failed to get int value, expr = %s",
                 af::SymbolicUtils::ToString(repeat_).c_str());
  const auto limited_size = kTransposeNoNeedUBConvertSize / GetSizeByDataType(transpose_node_->inputs[0].attr.dtype);
  score = dim < limited_size ? 1 : -1;
  return af::SUCCESS;
}
}  // namespace optimize
