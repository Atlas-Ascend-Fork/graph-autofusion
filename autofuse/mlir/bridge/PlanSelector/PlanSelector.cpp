/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelector.h"

#include <utility>

#include "bridge/BridgeDiagnostic.h"
#include "bridge/AttSchemaAdapter/AttSchemaAdapter.h"

namespace mlir::bridge {
bool SelectDynamicCandidates(const ascir::FusedScheduledResult &schedule, std::vector<PlanSelection> &selections,
                             std::string *diagnostic) {
  return BuildAllCandidateSelections(schedule, selections, diagnostic);
}

bool PreparePlanGeneration(const PlanGenerationRequest &request, PlanGenerationResult &result, std::string *diagnostic,
                           bool *unsupported) {
  result = {};
  if (unsupported) {
    *unsupported = false;
  }
  if (!request.resources.valid()) {
    SetDiagnostic(diagnostic, "plan generation requires valid target core and UB capacities");
    return false;
  }
  PlanGenerationResult prepared;
  if (!BuildAllCandidateSelections(request.schedule, prepared.selections, diagnostic)) {
    return false;
  }
  common::TilingFieldSchema schema;
  if (!common::BuildTilingFieldSchema(request.schedule, schema, diagnostic)) {
    return false;
  }
  auto options = request.options;
  options["tiling_data_type_name"] = "AutofuseTilingData";
  const std::string name = request.schedule.fused_graph_name.GetString();
  TilingCodeBundle program;
  if (RunAttForCandidates({name, request.schedule, prepared.selections, schema, options, request.is_inductor_scene},
                          program, diagnostic, unsupported) != af::SUCCESS) {
    return false;
  }
  // Static shapes still need ATT to choose tile sizes and a plan. Keep the
  // complete program for the host tiling library instead of compiling and
  // executing a separate solver during code generation.
  prepared.tiling = std::move(program);
  result = std::move(prepared);
  return true;
}

}  // namespace mlir::bridge
