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
#include "axes_solver_resources.h"

namespace att {
std::string GetAxesSolverSolverHead(bool enable_equal_order_tiling) {
  return enable_equal_order_tiling ? axes_resource::kEqualHead : axes_resource::kNormalHead;
}

std::string GetAxesSolverSolverFunc(bool enable_equal_order_tiling) {
  return enable_equal_order_tiling ? axes_resource::kEqualBody : axes_resource::kNormalBody;
}

std::string GetAxesSolverPgoSolverHead(int64_t pgo_step_max) {
  std::string codes;
  codes += "class AxesReorderPgoSolver : public AxesReorderSolver {\n";
  codes += "public:\n";
  codes += "  explicit AxesReorderPgoSolver(const AxesReorderSolverInput &input) : AxesReorderSolver(input) {}\n";
  codes += "  ~AxesReorderPgoSolver() = default;\n";
  codes += "  bool PgoSolverGenerateAllTilingData();\n";
  codes += "  std::vector<std::vector<uint32_t>> GetTilingDataList() { return availiable_tiling_data_list_; }\n";
  codes += "private:\n";
  codes += "  std::vector<std::vector<uint32_t>> availiable_tiling_data_list_;\n";
  codes += "  int64_t pgo_step_max_{" + std::to_string(pgo_step_max) + "};\n";
  codes += "  void PgoSolverGenerateAllTilingDataInner(const uint32_t index, std::vector<uint32_t> &ans_item,\n";
  codes +=
      "                                           std::vector<std::vector<uint32_t>> &ans, int64_t step_max = 16);\n";
  codes += "};\n";
  return codes;
}

std::string GetAxesSolverPgoSolverFunc() {
  return GenPgoSolverGenerateAllTilingData();
}

const std::string AXES_SOLVER_CODE_HEAD = GetAxesSolverSolverHead(true);
const std::string AXES_SOLVER_CODE_FUNC = GetAxesSolverSolverFunc(true);
const std::string AXES_SOLVER_PGO_CODE_FUNC = GetAxesSolverPgoSolverFunc();
}  // namespace att
