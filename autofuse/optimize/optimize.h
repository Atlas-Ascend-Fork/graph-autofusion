/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_OPTIMIZE_H_
#define OPTIMIZE_OPTIMIZE_H_

#include "ascir.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"
#include "ascgen_log.h"

namespace optimize {
namespace autoschedule {
struct AutoScheduleOutput;
}  // namespace autoschedule
enum class GraphType {
  kAscGraph = 0,     // ge
  kFusedAscBackend,  // inductor
  kFusedAscGraph,    // concat
  kInvalidGraph,
};
struct OptimizerOptions {
  GraphType graph_type = GraphType::kAscGraph;
};

enum class ReduceTemplateType : int32_t {
  kDefault = -1,  // 非reduce
  kCommon,        // 通用模板
  kAllLoad,       // 全载模板
  kRCore,         // R轴切多核模板
};

struct ScheduleTask {
  ::ascir::ImplGraph optimize_graph;
  std::vector<::ascir::ImplGraph> grouped_graphs;
  std::string score_func;
  std::map<size_t, std::vector<size_t>> groups_relations_in{};
  ReduceTemplateType reduce_type{ReduceTemplateType::kDefault};
  ::ascir::CubeTemplateType cube_type{::ascir::CubeTemplateType::kDefault};
  bool has_load_store_conversion{false};
  std::string semantic_name{};
};

class Optimizer {
 public:
  explicit Optimizer(const OptimizerOptions &options);

  /***
   *对fused_graph做前处理、auto schedule、内存分配
   */
  Status Optimize(const af::ComputeGraphPtr &fused_graph, ::ascir::FusedScheduledResult &fused_scheduled_result);
  /***
   *对hint_graph做前处理、auto schedule、内存分配
   */
  Status Optimize(af::AscGraph &hint_graph, ::ascir::FusedScheduledResult &fused_scheduled_result);

  void SetOptimizerOptions(const OptimizerOptions &options) {
    options_ = options;
  };

 private:
  /**
   * 对单张图做优化以及schedule
   * @param hint_graph ascgraph
   * @param scheduled_results 单张ascgraph的schedule返回值
   * @return
   */
  Status OptimizeForHintGraph(af::AscGraph &hint_graph, std::vector<::ascir::ScheduledResult> &scheduled_results) const;
  /**
   * 对单张图做优化以及schedule
   * @param fused_graph 带ascgraph或者ascbackend节点的计算图
   * @param fused_scheduled_result fusedgraph的schedule返回值
   * @return
   */
  Status OptimizeFusedAscBackend(const af::ComputeGraphPtr &fused_graph,
                                 ::ascir::FusedScheduledResult &fused_scheduled_result) const;
  Status DeserializeAscGraphNodes(const af::ComputeGraphPtr &fused_graph,
                                  std::map<af::Node *, af::AscGraph> &asc_backend_to_ascgraph,
                                  SizeVarSet &original_var_set, std::vector<af::Expression> &frontend_shape_vars);
  Status CollectAscBackendNodes(const af::ComputeGraphPtr &fused_graph,
                                std::map<af::Node *, af::AscGraph> &asc_backend_to_ascgraph,
                                SizeVarSet &original_var_set, std::vector<af::Expression> &frontend_shape_vars) const;

  /**
   * Buf/Que 分配
   * @param [in] graph 原始图
   * @param [in,out] impl_graphs schedule后的图，同时也将内存分配设置到这些图上
   */
  Status BufQueAlloc(const ::ascir::HintGraph &graph, std::vector<::ascir::ImplGraph> &impl_graphs) const;
  Status BufQueAlloc(const ::ascir::HintGraph &graph, ::ascir::ImplGraph &impl_graph) const;

  Status GraphPass(::ascir::ImplGraph &impl_graph) const;
  static Status RemoveAllZeroStrideLoopAxis(::ascir::ImplGraph &owner_graph);
  static Status MergeContinuousAxis(::ascir::ImplGraph &impl_graph,
                                    ::ascir::CubeTemplateType cube_type = ::ascir::CubeTemplateType::kDefault);
  // 一些算子再内存是连续的，但是合轴时需要当成非连续去处理
  static Status GetNonContinuousAxisPairBySpecialRule(::ascir::ImplGraph &impl_graph,
                                                      std::set<std::pair<int64_t, int64_t>> &non_continuous_pair);
  static bool IsReduceFirstStage(size_t index, ScheduleTask &schedule_task);
  Status ExpandReduceFirstStageResults(std::vector<autoschedule::AutoScheduleOutput> &schedule_outputs,
                                       std::vector<::ascir::ScheduledResult> &scheduled_results_cur, size_t index,
                                       ScheduleTask &schedule_task) const;
  void RefreshGroupRelation(size_t index, std::map<std::string, af::Expression> &var_relations,
                            ScheduleTask &schedule_task, ::ascir::ScheduledResult &schedule_result) const;
  static Status InitializeScheduledResults(std::vector<::ascir::ScheduledResult> &scheduled_results_cur,
                                           ScheduleTask &schedule_task);
  Status AutoScheduler(const ::ascir::HintGraph &hint_graph, ScheduleTask &schedule_task,
                       std::vector<::ascir::ScheduledResult> &scheduled_results) const;
  static void TryEnableGroupParallel(::ascir::FusedScheduledResult &fused_scheduled_result);
  static Status LoadOpSeqAdjust(const af::AscGraph &impl_graph);
  static void ExecSeqAdvancedOfLoad(const ::ascir::FusedScheduledResult &fused_scheduled_result);
  OptimizerOptions options_;
};
}  // namespace optimize

#endif  // OPTIMIZE_OPTIMIZE_H_
