/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_ARTIFACT_TILINGSOURCE_TILINGPROGRAMSPEC_H
#define AUTOFUSE_MLIR_ARTIFACT_TILINGSOURCE_TILINGPROGRAMSPEC_H

#include <cstdint>
#include <string>
#include <vector>

#include "dialect/AFIR/IR/AFIR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::bridge {

struct TilingTemplateSpec {
  int64_t template_id = -1;
  afir::ExprAttr workspace_size;
};

struct TilingStageSpec {
  int64_t stage_id = -1;
  std::string tiling_field_name;
  std::vector<std::string> fields;
  std::vector<TilingTemplateSpec> templates;
  // ATT derives only these destination variables from upstream stage tiling data.
  std::vector<std::string> derived_shape_vars;
};

struct TilingPlanSpec {
  int64_t subgraph_id = -1;
  int64_t plan_id = -1;
  std::vector<TilingStageSpec> stages;
};

struct TilingProgramSpec {
  std::vector<std::string> shape_params;
  std::vector<TilingPlanSpec> plans;
  // 参与聚合的候选共用的目标；其中的资源上限发射成 host tiling 里的 `g_no_limit_res`。
  afir::TargetAttr target;
};

// Builds the host-tiling contract exclusively from canonical metadata on the
// lowered Plan modules. Schedule-specific adaptation ends at ScheduleToAfir.
FailureOr<TilingProgramSpec> buildAggregateTilingProgramSpec(ArrayRef<ModuleOp> plan_modules);

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_ARTIFACT_TILINGSOURCE_TILINGPROGRAMSPEC_H
