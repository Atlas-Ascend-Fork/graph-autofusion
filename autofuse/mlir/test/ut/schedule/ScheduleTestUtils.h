/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TEST_UT_SCHEDULE_SCHEDULE_TEST_UTILS_H
#define AUTOFUSE_MLIR_TEST_UT_SCHEDULE_SCHEDULE_TEST_UTILS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "common/schedule_result.h"

namespace mlir::test {

inline ascir::ScheduledResult MakeScheduledResult(size_t template_count = 1U, bool empty_stage = false,
                                                  bool add_size_var = false) {
  ascir::ScheduledResult result;
  ascir::ScheduleGroup group;
  if (!empty_stage) {
    for (size_t template_id = 0U; template_id < template_count; ++template_id) {
      group.impl_graphs.emplace_back(("test_graph_" + std::to_string(template_id)).c_str());
      if (add_size_var) {
        group.impl_graphs.back().CreateSizeVar("s0");
      } else {
        // Initialize graph attributes, including ATT's unset tiling key (-1).
        group.impl_graphs.back().CreateSizeVar(1);
      }
    }
  }
  result.schedule_groups.push_back(std::move(group));
  return result;
}

inline ascir::FusedScheduledResult MakeFusedResult(size_t plan_count = 1U, size_t template_count = 1U,
                                                   bool empty_stage = false, bool add_size_var = false) {
  ascir::FusedScheduledResult result;
  std::vector<ascir::ScheduledResult> plans;
  for (size_t plan_id = 0U; plan_id < plan_count; ++plan_id) {
    plans.push_back(MakeScheduledResult(template_count, empty_stage, add_size_var));
  }
  result.node_idx_to_scheduled_results.push_back(std::move(plans));
  return result;
}

// Builds one subgraph per entry of `plan_counts_per_subgraph`, each holding
// that many plans with `template_count` templates in their single stage.
inline ascir::FusedScheduledResult MakeMultiSubgraphFusedResult(const std::vector<size_t> &plan_counts_per_subgraph,
                                                                size_t template_count = 1U) {
  ascir::FusedScheduledResult result;
  for (size_t plan_count : plan_counts_per_subgraph) {
    std::vector<ascir::ScheduledResult> plans;
    for (size_t plan_id = 0U; plan_id < plan_count; ++plan_id) {
      plans.push_back(MakeScheduledResult(template_count));
    }
    result.node_idx_to_scheduled_results.push_back(std::move(plans));
  }
  return result;
}

// Builds one plan with `stage_count` stages, each with `template_count`
// templates. Every stage reuses the SAME size-var name ("s0") and, when
// `reuse_queue_id` is true, the same Tque id 0 on every template's output --
// this is deliberate: legacy ProcessMultiGroup isolates each stage's fields
// into its own nested struct, so reusing small sequential queue/size-var
// names across stages is the common case, not an edge case. A schema builder
// that dedups by bare name instead of by (plan, stage) scope would collapse
// stage 0's and stage 1's distinct fields into one, silently discarding
// stage 1's tiling data.
inline ascir::ScheduledResult MakeMultiStageScheduledResult(size_t stage_count, size_t template_count = 1U) {
  ascir::ScheduledResult result;
  for (size_t stage_id = 0U; stage_id < stage_count; ++stage_id) {
    ascir::ScheduleGroup group;
    for (size_t template_id = 0U; template_id < template_count; ++template_id) {
      group.impl_graphs.emplace_back(
          ("test_graph_stage" + std::to_string(stage_id) + "_v" + std::to_string(template_id)).c_str());
      group.impl_graphs.back().CreateSizeVar("s0");
    }
    result.schedule_groups.push_back(std::move(group));
  }
  return result;
}

// One plan, one stage, one template graph carrying a single CONSTANT size var
// (via the af::AscGraph::CreateSizeVar(int64_t) overload). Used to verify
// that constant size vars are excluded from both the tiling schema and the
// schedule fingerprint -- they never become TilingData fields, so a
// fingerprint that included them would depend on information that is not
// actually part of the schedule's runtime-visible structure.
inline ascir::FusedScheduledResult MakeFusedResultWithConstSizeVar(int64_t const_value) {
  ascir::FusedScheduledResult result;
  ascir::ScheduledResult plan;
  ascir::ScheduleGroup group;
  af::AscGraph graph("test_graph_const");
  graph.CreateSizeVar(const_value);
  group.impl_graphs.push_back(std::move(graph));
  plan.schedule_groups.push_back(std::move(group));
  std::vector<ascir::ScheduledResult> plans;
  plans.push_back(std::move(plan));
  result.node_idx_to_scheduled_results.push_back(std::move(plans));
  return result;
}

inline ascir::FusedScheduledResult MakeMultiStageFusedResult(size_t plan_count, size_t stage_count,
                                                             size_t template_count = 1U) {
  ascir::FusedScheduledResult result;
  std::vector<ascir::ScheduledResult> plans;
  for (size_t plan_id = 0U; plan_id < plan_count; ++plan_id) {
    plans.push_back(MakeMultiStageScheduledResult(stage_count, template_count));
  }
  result.node_idx_to_scheduled_results.push_back(std::move(plans));
  return result;
}

}  // namespace mlir::test

#endif  // AUTOFUSE_MLIR_TEST_UT_SCHEDULE_SCHEDULE_TEST_UTILS_H
