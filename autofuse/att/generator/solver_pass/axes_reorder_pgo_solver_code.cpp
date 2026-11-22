/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generator/solver_pass/axes_reorder_solver_code.h"

namespace att {

// ============================================================================
// Section 7: PGO and Auto-tuning
// ============================================================================

std::string GenPgoSolverGenerateAllTilingDataHead() {
  std::string codes = R"(bool AxesReorderPgoSolver::PgoSolverGenerateAllTilingData() {
#if defined(AF_AXES_EXPLICIT_EVAL) && AF_AXES_EXPLICIT_EVAL
  eval_context_.ResetError();
  try {
#endif
  uint32_t index = 0;
  std::vector<uint32_t> ans_item(input_.tiling_vars_size);
  std::vector<std::vector<uint32_t>> ans;
  OP_LOGI(OP_NAME, "Start PgoSolverGenerate AllTilingData");
  int64_t step_max = 16;
  auto step_value = pgo_step_max_;
  if ((step_value % 2) == 0) {
    step_max = step_value;
  }
  PgoSolverGenerateAllTilingDataInner(index, ans_item, ans, step_max);
  availiable_tiling_data_list_ = ans;
  OP_LOGI(OP_NAME, "Gen PgoSolverGenerate AllTilingData %ld.", ans.size());
  return true;
#if defined(AF_AXES_EXPLICIT_EVAL) && AF_AXES_EXPLICIT_EVAL
  } catch (const EvalAbort &) {
    availiable_tiling_data_list_.clear();
    return false;
  }
#endif
}
)";
  return codes;
}

std::string GenPgoSolverGenerateAllTilingDataBody() {
  std::string codes = R"(void AxesReorderPgoSolver::PgoSolverGenerateAllTilingDataInner(const uint32_t index,
    std::vector<uint32_t> &ans_item, std::vector<std::vector<uint32_t>> &ans, int64_t step_max) {
  if (index >= input_.tiling_vars_size) {
    if (!SatisfyMCCons()) {
      return;
    }
    ans.push_back(ans_item);
    return;
  }
  TilingVariable* tilingDataVar;
  bool from_local_buffer_vars = true;
  if (index >= input_.local_buffer_vars_size) {
    tilingDataVar = input_.pure_mc_vars[index - input_.local_buffer_vars_size];
    from_local_buffer_vars = false;
  } else {
    tilingDataVar = input_.ordered_local_buffer_vars[index];
    from_local_buffer_vars = true;
  }
  auto min_ = tilingDataVar->align;
  auto max_ = CeilDiv(EvaluateUpperBound(tilingDataVar), tilingDataVar->align) * tilingDataVar->align;
  auto step = tilingDataVar->align;
  tilingDataVar->value = 0;
  auto var_value = tilingDataVar->value;
  OP_LOGI(OP_NAME, "PgoSolverGenerateAllTilingDataInner %zu.", ans.size());
)";
  return codes;
}

std::string GenPgoSolverGenerateAllTilingDataTail() {
  std::string codes = R"(  while (var_value < max_) {
    if (from_local_buffer_vars) {
      if (step <= step_max && tilingDataVar->value < step_max) {
        tilingDataVar->value = step;
      } else {
        tilingDataVar->value += step;
      }
      var_value = tilingDataVar->value;
      tilingDataVar->value = std::min(max_, tilingDataVar->value);
      step = std::min(step_max, step * 2);
    } else {
      tilingDataVar->value += step;
      var_value = tilingDataVar->value;
      if (tilingDataVar->value > max_) {
        continue;
      }
      step = 1;
    }
    // init 后面的变量
    auto tmp = index + 1;
    while (tmp < input_.local_buffer_vars_size + input_.pure_mc_vars_size) {
      if (tmp >= input_.local_buffer_vars_size) {
        auto tilingDataVarTmp = input_.pure_mc_vars[tmp - input_.local_buffer_vars_size];
        tilingDataVarTmp->value = tilingDataVarTmp->align;
      } else {
        auto tilingDataVarTmp = input_.ordered_local_buffer_vars[tmp];
        tilingDataVarTmp->value = tilingDataVarTmp->align;
      }
      tmp += 1;
    }

    if (!from_local_buffer_vars) {
      if (!SatisfyCons(tilingDataVar, ConstraintType::MC_MIXED)) {
        continue;
      }
    } else {
      if (!SatisfyCons(tilingDataVar, ConstraintType::LB_MIXED) || !SatisfyCons(tilingDataVar, ConstraintType::LOCAL_BUFFER)) {
        continue;
      }
    }
    if (from_local_buffer_vars) {
      ans_item[input_.ordered_to_canonical[index]] = tilingDataVar->value;
    } else {
      ans_item[index] = tilingDataVar->value;
    }
    PgoSolverGenerateAllTilingDataInner(index + 1, ans_item, ans, step_max);
  }
}
)";
  return codes;
}

std::string GenPgoSolverGenerateAllTilingData() {
  std::string TilingDataHead = GenPgoSolverGenerateAllTilingDataHead();
  std::string TilingDataBody = GenPgoSolverGenerateAllTilingDataBody();
  std::string TilingDataTail = GenPgoSolverGenerateAllTilingDataTail();
  return TilingDataHead + TilingDataBody + TilingDataTail;
}

}  // namespace att
