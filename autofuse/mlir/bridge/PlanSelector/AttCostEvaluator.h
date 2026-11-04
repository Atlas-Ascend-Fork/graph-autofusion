/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_ATT_COST_EVALUATOR_H
#define AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_ATT_COST_EVALUATOR_H

#include "bridge/PlanSelector/PlanSelector.h"
#include "base/model_info.h"

namespace mlir::bridge::detail {

// Internal ATT cost evaluation, separated from model generation so selection
// can be tested with explicit models. These functions do not query ATT.
bool ScoreModel(const att::ModelInfo &model, const ConcreteShape &shape, long double &score);

// Score a structurally valid schedule. A plan is eligible only when every
// stage/template has a matching model and a constant finite cost. Missing or
// unevaluable plans are skipped; every subgraph must still have a winner.
// On failure, selections is empty.
bool SelectWinnersFromModels(const ascir::FusedScheduledResult &fused_result,
                             const att::FusedParsedScheduleResult &models, const ConcreteShape &shape,
                             std::vector<PlanSelection> &selections, std::string *diagnostic);

}  // namespace mlir::bridge::detail

#endif  // AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_ATT_COST_EVALUATOR_H
