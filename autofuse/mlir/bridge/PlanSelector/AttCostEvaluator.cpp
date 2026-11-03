/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/AttCostEvaluator.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "bridge/BridgeDiagnostic.h"

namespace mlir::bridge::detail {
namespace {

bool EvaluateCost(const af::Expression &expression, const ConcreteShape &shape, long double &value) {
  std::vector<std::pair<af::Expression, af::Expression>> substitutions;
  substitutions.reserve(shape.size());
  for (const auto &[name, dimension] : shape) {
    substitutions.emplace_back(af::Expression::Parse(name.c_str()),
                               af::Expression::Parse(std::to_string(dimension).c_str()));
  }
  const af::Expression substituted = expression.Subs(substitutions).Simplify();
  if (!substituted.IsConstExpr()) {
    return false;
  }
  const auto text = substituted.Str();
  if (text == nullptr) {
    return false;
  }
  const std::string costText(text.get());
  size_t parsedSize = 0;
  try {
    const long double parsed = std::stold(costText, &parsedSize);
    if (parsedSize != costText.size() || !std::isfinite(parsed)) {
      return false;
    }
    value = parsed;
    return true;
  } catch (const std::invalid_argument &) {
    return false;
  } catch (const std::out_of_range &) {
    return false;
  }
}

}  // namespace

bool ScoreModel(const att::ModelInfo &model, const ConcreteShape &shape, long double &score) {
  // Hardware pipes run in parallel; stages run in sequence. ATT's solver
  // uses max(pipe costs) + head_cost for each stage.
  long double max_pipe_cost = 0.0L;
  bool has_pipe_cost = false;
  for (const auto &[_, expression] : model.objects) {
    long double term = 0.0L;
    if (!EvaluateCost(expression, shape, term)) {
      return false;
    }
    if (!has_pipe_cost || term > max_pipe_cost) {
      max_pipe_cost = term;
    }
    has_pipe_cost = true;
  }
  long double head_cost = 0.0L;
  if (!EvaluateCost(model.head_cost, shape, head_cost)) {
    return false;
  }
  score = (has_pipe_cost ? max_pipe_cost : 0.0L) + head_cost;
  return std::isfinite(score);
}

namespace {

bool SelectBestTemplate(const std::vector<att::ModelInfo> &models, const ascir::ScheduleGroup &group,
                        const ConcreteShape &shape, size_t &template_id, long double &score) {
  if (models.size() != group.impl_graphs.size() || models.empty()) {
    return false;
  }
  score = std::numeric_limits<long double>::max();
  bool found = false;
  for (size_t candidate_id = 0U; candidate_id < models.size(); ++candidate_id) {
    long double candidate_score = 0.0L;
    if (!detail::ScoreModel(models[candidate_id], shape, candidate_score)) {
      return false;
    }
    if (!found || candidate_score < score) {
      score = candidate_score;
      template_id = candidate_id;
      found = true;
    }
  }
  return found;
}

bool SelectTemplatesForPlan(const ascir::ScheduledResult &plan, const att::ParsedScheduleResult &model,
                            const ConcreteShape &shape, PlanSelection &selection, long double &score) {
  if (plan.schedule_groups.empty() || model.groups_tiling_model_info.size() != plan.schedule_groups.size()) {
    return false;
  }
  selection.template_ids_per_stage.resize(plan.schedule_groups.size());
  score = 0.0L;
  for (size_t stage_id = 0U; stage_id < plan.schedule_groups.size(); ++stage_id) {
    const auto model_group = model.groups_tiling_model_info.find(stage_id);
    if (model_group == model.groups_tiling_model_info.end()) {
      return false;
    }
    size_t template_id = 0U;
    long double stage_score = 0.0L;
    if (!SelectBestTemplate(model_group->second, plan.schedule_groups[stage_id], shape, template_id, stage_score)) {
      return false;
    }
    selection.template_ids_per_stage[stage_id].push_back(template_id);
    score += stage_score;
  }
  return std::isfinite(score);
}

bool SelectBestPlanForSubgraph(const std::vector<ascir::ScheduledResult> &plans,
                               const std::map<size_t, att::ParsedScheduleResult> &model_plans,
                               const ConcreteShape &shape, size_t subgraph_id, PlanSelection &best_selection) {
  long double best_score = std::numeric_limits<long double>::max();
  bool found = false;
  for (size_t plan_id = 0U; plan_id < plans.size(); ++plan_id) {
    const auto model_plan = model_plans.find(plan_id);
    if (model_plan == model_plans.end()) {
      continue;
    }
    PlanSelection candidate;
    long double plan_score = 0.0L;
    if (SelectTemplatesForPlan(plans[plan_id], model_plan->second, shape, candidate, plan_score) &&
        (!found || plan_score < best_score)) {
      candidate.subgraph_id = subgraph_id;
      candidate.plan_id = plan_id;
      best_score = plan_score;
      best_selection = std::move(candidate);
      found = true;
    }
  }
  return found;
}

}  // namespace

bool SelectWinnersFromModels(const ascir::FusedScheduledResult &fused_result,
                             const att::FusedParsedScheduleResult &models, const ConcreteShape &shape,
                             std::vector<PlanSelection> &selections, std::string *diagnostic) {
  selections.clear();
  std::vector<PlanSelection> winners;
  winners.reserve(fused_result.node_idx_to_scheduled_results.size());
  for (size_t subgraph_id = 0U; subgraph_id < fused_result.node_idx_to_scheduled_results.size(); ++subgraph_id) {
    const auto model_subgraph = models.find(subgraph_id);
    if (model_subgraph == models.end()) {
      SetDiagnostic(diagnostic, "ATT returned no model information for subgraph " + std::to_string(subgraph_id));
      return false;
    }
    PlanSelection winner;
    if (!SelectBestPlanForSubgraph(fused_result.node_idx_to_scheduled_results[subgraph_id], model_subgraph->second,
                                   shape, subgraph_id, winner)) {
      SetDiagnostic(diagnostic,
                    "ATT did not provide a complete, constant-cost plan for subgraph " + std::to_string(subgraph_id));
      return false;
    }
    winners.push_back(std::move(winner));
  }
  selections = std::move(winners);
  return true;
}

}  // namespace mlir::bridge::detail
