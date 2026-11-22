/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelection.h"

#include <numeric>
#include <set>
#include <utility>

#include "bridge/BridgeDiagnostic.h"
#include "common/tiling_schema/template_keys.h"

namespace mlir::bridge {
namespace {

bool ValidatePlanStructure(const ascir::ScheduledResult &plan, size_t subgraph_id, size_t plan_id,
                           std::string *diagnostic) {
  const std::string location = "subgraph " + std::to_string(subgraph_id) + " plan " + std::to_string(plan_id);
  if (plan.schedule_groups.empty()) {
    SetDiagnostic(diagnostic, location + " contains no stage");
    return false;
  }
  for (const auto &stage : plan.schedule_groups) {
    if (stage.impl_graphs.empty()) {
      SetDiagnostic(diagnostic, location + " contains an empty stage");
      return false;
    }
    if (!common::ValidateTemplateKeys(stage, diagnostic)) {
      return false;
    }
  }
  return true;
}

std::vector<size_t> MakeContiguousIds(size_t count) {
  std::vector<size_t> ids(count);
  std::iota(ids.begin(), ids.end(), size_t{0U});
  return ids;
}

}  // namespace

bool ValidatePlanSelection(const ascir::FusedScheduledResult &fused_result, const PlanSelection &selection,
                           std::string *diagnostic) {
  if (selection.subgraph_id >= fused_result.node_idx_to_scheduled_results.size()) {
    SetDiagnostic(diagnostic, "subgraph_id is out of range");
    return false;
  }
  const auto &plans = fused_result.node_idx_to_scheduled_results[selection.subgraph_id];
  if (selection.plan_id >= plans.size()) {
    SetDiagnostic(diagnostic, "plan_id is out of range");
    return false;
  }
  const auto &plan = plans[selection.plan_id];
  if (!ValidatePlanStructure(plan, selection.subgraph_id, selection.plan_id, diagnostic)) {
    return false;
  }
  const auto &stages = plan.schedule_groups;
  if (selection.template_ids_per_stage.size() != stages.size()) {
    SetDiagnostic(diagnostic, "template_ids_per_stage does not cover every stage");
    return false;
  }
  for (size_t stage_id = 0U; stage_id < stages.size(); ++stage_id) {
    const auto &templates = selection.template_ids_per_stage[stage_id];
    const std::string location = " for stage " + std::to_string(stage_id);
    if (templates.empty()) {
      SetDiagnostic(diagnostic, "template_ids is empty" + location);
      return false;
    }
    std::set<size_t> seen;
    for (size_t template_id : templates) {
      if (template_id >= stages[stage_id].impl_graphs.size()) {
        SetDiagnostic(diagnostic, "template_id is out of range" + location);
        return false;
      }
      if (!seen.insert(template_id).second) {
        SetDiagnostic(diagnostic, "duplicate template_id" + location);
        return false;
      }
    }
  }
  return true;
}

bool BuildAllCandidateSelections(const ascir::FusedScheduledResult &fused_result,
                                 std::vector<PlanSelection> &selections, std::string *diagnostic) {
  selections.clear();
  const auto &subgraphs = fused_result.node_idx_to_scheduled_results;
  if (subgraphs.empty()) {
    SetDiagnostic(diagnostic, "fused schedule contains no candidate plan");
    return false;
  }
  std::vector<PlanSelection> candidates;
  for (size_t subgraph_id = 0U; subgraph_id < subgraphs.size(); ++subgraph_id) {
    const auto &plans = subgraphs[subgraph_id];
    if (plans.empty()) {
      SetDiagnostic(diagnostic, "subgraph " + std::to_string(subgraph_id) + " contains no candidate plan");
      return false;
    }
    for (size_t plan_id = 0U; plan_id < plans.size(); ++plan_id) {
      const auto &plan = plans[plan_id];
      if (!ValidatePlanStructure(plan, subgraph_id, plan_id, diagnostic)) {
        return false;
      }
      PlanSelection candidate{subgraph_id, plan_id, {}};
      for (const auto &stage : plan.schedule_groups) {
        candidate.template_ids_per_stage.push_back(MakeContiguousIds(stage.impl_graphs.size()));
      }
      candidates.push_back(std::move(candidate));
    }
  }
  selections = std::move(candidates);
  return true;
}

}  // namespace mlir::bridge
