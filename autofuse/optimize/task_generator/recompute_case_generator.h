/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_RECOMPUTE_CASE_GENERATOR_H_
#define ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_RECOMPUTE_CASE_GENERATOR_H_

#include "ascir.h"
#include "ascgen_log.h"
#include "schedule_task_generator.h"
#include "task_generator/schedule_case_generator.h"

namespace optimize {
class RecomputeCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate([[maybe_unused]] ascir::HintGraph &graph, [[maybe_unused]] std::vector<ascir::ImplGraph> &graphs,
                  [[maybe_unused]] std::vector<std::string> &score_functions) override {
    return af::GRAPH_SUCCESS;
  }
  Status GeneratorTask(ascir::HintGraph &hint_graph, std::vector<ScheduleTask> &tasks,
                       const OptimizerOptions &options) override;

 private:
  struct SplitResultInOnePath {
    std::unordered_set<af::Node *> recompute_nodes;  // 需要重计算的节点
    std::unordered_set<af::Node *> valid_path;       // 有效路径
    std::unordered_set<af::Node *> output_nodes;     // output节点
  };

  struct SplitResult {
    std::vector<SplitResultInOnePath> valid_path_results;   // 有效路径
    std::unordered_set<af::Node *> merged_recompute_nodes;  // 需要重计算的节点
    std::unordered_set<af::Node *> merged_output_nodes;     // output节点
    std::unordered_set<af::Node *> merged_path_nodes;       // 所有在拆分路径上的节点
  };

  bool IsRecomputableNode(ascir::HintGraph &hint_graph, af::AscNode *node) const;
  bool IsRecomputableAlwaysBetter() const;
  Status AnalyzeSplittablePath(ascir::HintGraph &hint_graph, const af::AscNodePtr &potential_store);
  void MergeAllPaths();
  Status GenerateBroadcastWorkspaceTask(ascir::HintGraph &hint_graph, std::vector<ScheduleTask> &tasks);
  Status CollectBroadcastCandidates(const ascir::HintGraph &hint_graph, ascir::ImplGraph &optimize_graph,
                                    std::vector<af::AscNodePtr> &node_order,
                                    std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &broadcast_candidates);
  Status SelectBroadcastPartitions(ascir::ImplGraph &optimize_graph,
                                   const std::vector<std::pair<af::AscNodePtr, af::AscNodePtr>> &broadcast_candidates,
                                   std::vector<af::AscNodePtr> &node_order, bool &partitioned);
  Status DoTaskGenerator(ascir::ImplGraph &impl_graph, std::vector<ScheduleTask> &tasks) const;
  Status DoGraphSplit(af::AscGraph &graph) const;

  bool is_static_graph_{false};
  SplitResult result_;
};

// Broadcast workspace 切分模板的打分函数生成器, 类比 ConcatScoreFunctionGenerator:
// 按原始图非首轴(除首个 axis 外) size 乘积与 32K 阈值比较决定打分: 超阈值返回正分使本模板在
// tiling 期与其他候选模板(默认模板/重计算模板, 空打分默认0分)竞争中优先; 未超阈值返回负分劣后给
// 默认模板——平分时 result 序号小者胜, 本模板恒为 result0(双平台均以 tasks.empty() 守卫
// RecomputeCaseGenerator), 返回 0 分无法落选。
// 动态 shape 场景生成运行时代码, 从 tiling data 嵌套结构读取 size var 实际值; 嵌套结构
// InputParams 按 schedule group 定义(仅含该 group 子图用到的 size var), 故 Generate 须在
// PartitionByConnectivity 切分完成后调用, 逐自由符号定位实际所在 group 后生成跨 group 引用。
class BroadcastWorkspaceScoreFunctionGenerator {
 public:
  BroadcastWorkspaceScoreFunctionGenerator(const ascir::HintGraph &graph,
                                           const std::vector<ascir::ImplGraph> &grouped_graphs);
  Status Generate(std::string &score_func);

 private:
  Status ParseNonFirstAxisProduct();
  Status GenerateForDynamic();
  void GenerateReturnValue(const int32_t score);

  const ascir::HintGraph *graph_;
  const std::vector<ascir::ImplGraph> *grouped_graphs_;
  af::Expression product_;
  std::stringstream ss_;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_RECOMPUTE_CASE_GENERATOR_H_
