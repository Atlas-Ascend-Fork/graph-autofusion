/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCHDULE_CASE_GENERATOR_H
#define REDUCE_SCHDULE_CASE_GENERATOR_H

#include "ascir.h"
#include "ascgen_log.h"
#include "ascir_ops.h"
#include "task_generator/schedule_case_generator.h"
#include <map>

namespace optimize {

using ReduceType = std::variant<af::ascir_op::Max, af::ascir_op::Sum, af::ascir_op::Min, af::ascir_op::Prod,
                                af::ascir_op::Any, af::ascir_op::All, af::ascir_op::ArgMaxMultiRPhase1,
                                af::ascir_op::ArgMaxMultiRPhase2, af::ascir_op::Softmax>;

class ReducePartitionCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;
  Status GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                       const OptimizerOptions &options) override;
  Status GenerateGeneralCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                             std::vector<std::string> &score_functions);
  Status GenerateAllLoadCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                             const std::vector<std::string> &score_functions);

 private:
  struct Citation {
    af::AscNodePtr source;
    af::AscNodePtr citation;
    af::AscNodePtr reduce;
  };
  using CitationGroups = std::vector<std::vector<Citation>>;

  Status GeneratorGeneralTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks);
  Status GeneratorAllLoadTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks);
  Status GeneratorRCoreTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks) const;
  Status ReducePartitionPostFusion(ascir::ImplGraph &impl_graph);
  Status GenerateBroadcastWorkspaceCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs);
  Status PartitionByNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph);
  bool IsInputNodePartitioned(const std::shared_ptr<af::Node> &start, const std::shared_ptr<af::Node> &node);
  Status FindNormLoop(const af::AscNodePtr &start, std::vector<af::AscNodePtr> &ends);
  bool IsNorm(const af::AscNodePtr &start, const af::AscNodePtr &end);
  std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> FindNormLoops(ascir::ImplGraph &graph);
  // 补充类 Norm 结构 loops: FindNormLoops 只沿输出方向从多输出节点展开, 这里在其结果
  // 基础上补充非单输出方向的类 Norm 结构(起点, 归宿点)对
  Status AddNormLikeLoops(ascir::ImplGraph &graph,
                          std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &loop_start_end);
  Status PartitionNorm(ascir::ImplGraph &impl_graph,
                       std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &loop_start_end);
  void FindAllPath(const af::AscNodePtr &start, const af::AscNodePtr &end, std::vector<af::AscNodePtr> &path,
                   std::vector<std::vector<af::AscNodePtr>> &all_paths);
  static Status PartitionLoadNode(af::AscNodePtr &src_load_node, af::AscNodePtr &dst_node,
                                  ascir::ImplGraph &impl_graph);
  static Status PartitionScalarNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph);
  static Status PartitionDataNode(af::AscNodePtr &src_node, af::AscNodePtr &dst_node, ascir::ImplGraph &impl_graph);
  static bool HasReduce(const ascir::ImplGraph &impl_graph);
  static bool HasArgMaxReduce(const ScheduleTask &task);
  static bool HasMultiRAxisReduce(const ascir::ImplGraph &impl_graph);
  static bool IsOnlyHasOneOrLessReduce(const ascir::ImplGraph &impl_graph);
  static bool CanFullLoadReduceFuse(const ascir::ImplGraph &impl_graph);
  Status ReducePartitionMultipleCitations(ascir::ImplGraph &impl_graph);
  // 公共动作: 反复摘除 PostFusion/PartitionNorm/多引用切分产生的悬空节点(无数据输出且
  // 非 Output/workspace 写侧), 在三个切分步骤完成后、子图划分前统一执行
  Status RemoveDanglingNodes(ascir::ImplGraph &impl_graph);
  Status CollectCitationGroups(ascir::ImplGraph &impl_graph, CitationGroups &citation_groups);
  void BuildCitationGroupAnchors(const CitationGroups &citation_groups, std::vector<size_t> &parent,
                                 std::map<size_t, af::AscNodePtr> &group_anchors);
  Status PartitionCitationGroups(ascir::ImplGraph &impl_graph, const CitationGroups &citation_groups,
                                 const std::vector<size_t> &parent,
                                 const std::map<size_t, af::AscNodePtr> &group_anchors);
  bool FindOutputReduce(const af::AscNodePtr &node, af::AscNodePtr &reduce_node);
  Status PartitionReduceNode(af::AscNodePtr &src_node, ascir::ImplGraph &impl_graph);
  static Status ExpandDimsForRA(ascir::ImplGraph &impl_graph);
  static bool IsRAStructure(const ascir::ImplGraph &impl_graph);

  bool partition_{false};
  std::vector<af::AscNodePtr> node_order_{};
};

class RMulticorePhase2Graph {
 public:
  RMulticorePhase2Graph(ascir::ImplGraph &phase2graph, ascir::ImplGraph &phase1graph, ascir::ImplGraph &phase_graph,
                        af::AscNodePtr reduce_node)
      : phase2graph(phase2graph),
        phase1graph(phase1graph),
        phase_graph(phase_graph),
        reduce_node(std::move(reduce_node)) {};
  RMulticorePhase2Graph(const RMulticorePhase2Graph &) = delete;
  RMulticorePhase2Graph &operator=(const RMulticorePhase2Graph &) = delete;
  Status Construct();

 private:
  af::Expression Rm_org_size;
  af::Expression A_org_size;
  ascir::ImplGraph &phase2graph;
  ascir::ImplGraph &phase1graph;
  ascir::ImplGraph &phase_graph;
  af::AscNodePtr reduce_node;
  Status CreateVarAxis();
  Status CompleteNodeAttr(af::AscNodePtr &node, bool before_reduce, const af::AscTensorDataType &data_type);
  Status CompleteNodeAttrBeforeReduce(af::AscNodePtr &node);
  Status CompletePhaseGraph(ReduceType &phase2graph_reduce);
  Status PartitionByReduce(ascir::ImplGraph &impl_graph, ReduceType &phase2graph_reduce,
                           std::vector<af::AscNodePtr> &node_order);
  Status SetNodeOrder(std::vector<af::AscNodePtr> &node_order);
  Status SetupArgMaxIndexNodes(const af::AscNodePtr &reduce_node, ascir::ImplGraph &phase2graph);
};
}  // namespace optimize

#endif  // REDUCE_SCHDULE_CASE_GENERATOR_H
