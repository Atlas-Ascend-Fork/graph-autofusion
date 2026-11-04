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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "bridge/PlanSelector/PlanSelection.h"

namespace mlir::bridge {

using ConcreteShape = std::map<std::string, int64_t>;
using AttOptions = std::map<std::string, std::string>;

// Retain every plan and every stage template, leaving the choice to the ATT
// solver at run time (it dispatches on `graph<N>_tiling_key`).
//
// This function only builds selections; it does not generate tiling or dispatch.
// `unsupported` 的语义与 `BuildAllCandidateSelections` 的那一位逐字相同（见 PlanSelection.h）：
// 这两个入口只是把它原样透出来。
bool SelectDynamicCandidates(const ascir::FusedScheduledResult &fused_result, std::vector<PlanSelection> &selections,
                             std::string *diagnostic = nullptr, bool *unsupported = nullptr);

// Retain one plan per subgraph and one template per stage for a static shape.
// If every subgraph has a unique candidate, return it without ATT model generation.
// Otherwise use ATT cost models; unresolved/non-finite scores cannot win, and
// ties retain the first candidate in input order. An empty shape map is valid
// when no symbol substitution is needed. Tiling generation remains downstream.
// If no complete winner can be determined, retain every candidate for runtime selection.
// Both selectors reject malformed schedule structure and clear selections on failure.
bool SelectStaticWinner(const ascir::FusedScheduledResult &fused_result, const ConcreteShape &shape,
                        const AttOptions &options, std::vector<PlanSelection> &selections,
                        std::string *diagnostic = nullptr, bool *unsupported = nullptr);

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTOR_H
