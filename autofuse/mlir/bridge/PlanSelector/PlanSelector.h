/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTOR_H
#define AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTOR_H
#include <map>
#include <string>
#include <vector>
#include "bridge/PlanSelector/PlanSelection.h"
#include "bridge/AttSchemaAdapter/TilingCodeBundle.h"
namespace mlir::bridge {
using ConcreteShape = std::map<std::string, int64_t>;
enum class PlanGenerationMode { Static, Dynamic };
struct PlanGenerationRequest {
  const ascir::FusedScheduledResult &schedule;
  PlanGenerationMode mode;
  common::ResourceBudget resources;
  bool is_inductor_scene;
  ConcreteShape shape;
  AttOptions options;
};
// selections and tiling are committed together only after successful preparation.
struct PlanGenerationResult {
  std::vector<PlanSelection> selections;
  TilingArtifact tiling;
};
bool SelectDynamicCandidates(const ascir::FusedScheduledResult &schedule, std::vector<PlanSelection> &selections,
                             std::string *diagnostic = nullptr);
bool PreparePlanGeneration(const PlanGenerationRequest &request, PlanGenerationResult &result,
                           std::string *diagnostic = nullptr, bool *unsupported = nullptr);
}  // namespace mlir::bridge
#endif
