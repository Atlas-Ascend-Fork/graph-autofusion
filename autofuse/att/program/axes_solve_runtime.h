/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_PROGRAM_AXES_SOLVE_RUNTIME_H_
#define ATT_PROGRAM_AXES_SOLVE_RUNTIME_H_
#include "serialization/axes_evaluation_codec.h"

namespace att {
// Fingerprint of the exact common solver and scalar helpers used by this build.
const char *GetAxesSolverSourceIdentityV1();
enum class AxesSolveOutcome { kSolved, kEmpty, kNoSolution };
struct AxesSolveOptions {
  bool trade_off = false;
  bool tune_block = false;
  bool tune = false;
  bool equal_order = false;
};

struct AxesSolveResources {
  uint32_t core_num = 0;
  uint32_t ub_size = 0;
  double ub_threshold = 0.2f;
  double corenum_threshold = 0.4f;
  double perf_threshold = 0.0f;
};

// Internal categories are mapped explicitly from the shared solver status.
enum class AxesSolveFailureStatus : uint32_t {
  kSuccess = 0,
  kInvalidBinding,
  kTypeMismatch,
  kArithmeticError,
  kSdkError,
  kOther
};

// Per-call internal diagnostic. UINT32_MAX means that a location is unavailable.
// It is written only for the first evaluation failure in one invocation.
struct AxesSolveEvaluationFailure {
  uint32_t status = 0;
  uint32_t expression_id = UINT32_MAX;
  uint32_t stage_step = UINT32_MAX;
  uint32_t instruction = UINT32_MAX;
  bool derived = false;
  uint32_t derived_stage = UINT32_MAX;
  uint32_t cache_index = UINT32_MAX;
};

// Internal normal-order entry used by ATT. No native solver pointers cross this boundary.
TilingExpressionStatus RunAxesSolveRecord(const AxesSolveRecords &records, const std::vector<int64_t> &shapes,
                                          const AxesSolveResources &resources, const AxesSolveOptions &options,
                                          void *destination, size_t capacity, AxesSolveOutcome *outcome,
                                          const AxesProgramScratch *scratch = nullptr, uint32_t *block_dim = nullptr);

// Extended internal entry. The original symbol remains available unchanged.
__attribute__((visibility("hidden"))) TilingExpressionStatus RunAxesSolveRecordWithFailure(
    const AxesSolveRecords &records, const std::vector<int64_t> &shapes, const AxesSolveResources &resources,
    const AxesSolveOptions &options, void *destination, size_t capacity, AxesSolveOutcome *outcome,
    const AxesProgramScratch *scratch, uint32_t *block_dim, AxesSolveEvaluationFailure *failure);
}  // namespace att
#endif
