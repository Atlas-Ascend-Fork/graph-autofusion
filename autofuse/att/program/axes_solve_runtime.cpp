/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <any>
#include <cmath>
#include <functional>
#include <sstream>
#include "framework/common/debug/ge_log.h"
#include "program/axes_solve_adapter.h"
#include "axes_solver_resources.h"

namespace att {
const char *GetAxesSolverSourceIdentityV1() {
  return axes_resource::kSourceSha256;
}

namespace axes_runtime {
#define AF_AXES_EQUAL_ORDER 0
#define AF_AXES_EXPLICIT_EVAL 1
#define OP_LOGD(name, format, ...) GELOGD("%s: " format, name, ##__VA_ARGS__)
#define OP_LOGI(name, format, ...) GELOGI("%s: " format, name, ##__VA_ARGS__)
#define OP_LOGW(name, format, ...) GELOGW("%s: " format, name, ##__VA_ARGS__)
constexpr char OP_NAME[] = "AxesProgram";
#include "core/scalar/expression_helpers.inc"
#include "core/scalar/scalar_helpers.inc"
#include "core/scalar/axes_ceil_div.inc"
#include "core/axes_reorder/axes_reorder_head.inc"
#include "core/axes_reorder/axes_reorder_body.inc"
#undef OP_LOGW
#undef OP_LOGI
#undef OP_LOGD
#undef AF_AXES_EXPLICIT_EVAL
#undef AF_AXES_EQUAL_ORDER
}  // namespace axes_runtime

TilingExpressionStatus RunAxesSolveRecord(const AxesSolveRecords &records, const std::vector<int64_t> &shapes,
                                          const AxesSolveResources &resources, const AxesSolveOptions &options,
                                          void *destination, size_t capacity, AxesSolveOutcome *outcome,
                                          const AxesProgramScratch *scratch, uint32_t *block_dim) {
  return RunAxesSolveRecordWithFailure(records, shapes, resources, options, destination, capacity, outcome, scratch,
                                       block_dim, nullptr);
}

TilingExpressionStatus RunAxesSolveRecordWithFailure(
    const AxesSolveRecords &records, const std::vector<int64_t> &shapes, const AxesSolveResources &resources,
    const AxesSolveOptions &options, void *destination, size_t capacity, AxesSolveOutcome *outcome,
    const AxesProgramScratch *scratch, uint32_t *block_dim, AxesSolveEvaluationFailure *failure) {
  if (options.equal_order) {
    return TilingExpressionStatus::kUnsupported;
  }
  if (resources.core_num == 0) {
    return TilingExpressionStatus::kInvalidArgument;
  }
  axes_runtime::AxesReorderSolverInput input;
  input.core_num = resources.core_num;
  input.ub_size = resources.ub_size;
  input.ub_threshold = resources.ub_threshold;
  input.corenum_threshold = resources.corenum_threshold;
  input.perf_threshold = resources.perf_threshold;
  return SolveAxesRecord<axes_runtime::AxesReorderSolver, axes_runtime::AxesReorderSolverInput,
                         axes_runtime::EvalContext, axes_runtime::EvalBinding, axes_runtime::TypedValue>(
      records, shapes, input, options, destination, capacity, outcome, scratch, block_dim, failure);
}
}  // namespace att
