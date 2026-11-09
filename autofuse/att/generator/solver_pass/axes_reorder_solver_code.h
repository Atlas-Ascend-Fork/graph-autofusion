/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_AXES_REORDER_SOLVER_CODE_H_
#define ATT_AXES_REORDER_SOLVER_CODE_H_
#include <cstdint>
#include <string>
namespace att {
std::string GetAxesSolverSolverHead(bool enable_equal_order_tiling = false);
std::string GetAxesSolverSolverFunc(bool enable_equal_order_tiling = false);
std::string GetAxesSolverPgoSolverHead(int64_t pgo_step_max);
std::string GetAxesSolverPgoSolverFunc();
std::string GenPgoSolverGenerateAllTilingData();
extern const std::string AXES_SOLVER_CODE_HEAD;
extern const std::string AXES_SOLVER_CODE_FUNC;
extern const std::string AXES_SOLVER_PGO_CODE_FUNC;
}  // namespace att
#endif  // ATT_AXES_REORDER_SOLVER_CODE_H_
