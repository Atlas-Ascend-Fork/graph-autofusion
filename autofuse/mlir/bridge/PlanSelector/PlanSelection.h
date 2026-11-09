/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTION_H
#define AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTION_H

#include <cstddef>
#include <string>
#include <vector>

#include "common/schedule_result.h"

namespace mlir::bridge {

// Coordinates in the input FusedScheduledResult. Each selection represents
// one complete plan, with the retained template ids for every stage.
struct PlanSelection {
  size_t subgraph_id{0U};
  size_t plan_id{0U};
  std::vector<std::vector<size_t>> template_ids_per_stage;

  bool operator==(const PlanSelection &other) const {
    return subgraph_id == other.subgraph_id && plan_id == other.plan_id &&
           template_ids_per_stage == other.template_ids_per_stage;
  }
};

// Check plan/stage structure and selection coordinates, not resource feasibility.
bool ValidatePlanSelection(const ascir::FusedScheduledResult &fused_result, const PlanSelection &selection,
                           std::string *diagnostic = nullptr);

// Enumerate every plan and stage template in input order. Only schedule structure
// is checked here; resource and implementation capabilities are checked downstream.
// Empty fusion/subgraph/plan/stage is an error; selections is empty on failure.
bool BuildAllCandidateSelections(const ascir::FusedScheduledResult &fused_result,
                                 std::vector<PlanSelection> &selections, std::string *diagnostic = nullptr);

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTION_H
