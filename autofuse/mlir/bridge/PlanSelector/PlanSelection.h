/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTION_H
#define AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTION_H

#include <cstddef>
#include <string>
#include <vector>

#include "common/schedule_result.h"

namespace mlir::bridge {

// Coordinates in the input FusedScheduledResult. Each selection represents
// one complete plan, with the retained template ids for every stage.
struct PlanSelection {
  size_t subgraph_id{0U};
  size_t plan_id{0U};
  std::vector<std::vector<size_t>> template_ids_per_stage;

  bool operator==(const PlanSelection &other) const {
    return subgraph_id == other.subgraph_id && plan_id == other.plan_id &&
           template_ids_per_stage == other.template_ids_per_stage;
  }
};

// Check plan/stage structure and selection coordinates, not resource feasibility.
bool ValidatePlanSelection(const ascir::FusedScheduledResult &fused_result, const PlanSelection &selection,
                           std::string *diagnostic = nullptr);

// Enumerate all plans and MLIR-supported stage templates in input order. A
// template is excluded when MLIR cannot lower its data movement, inline
// elementwise broadcast, or queue resource view. Empty fusion, subgraph, plan,
// stage, or a plan with no supported template is an error. On failure,
// selections is empty.
//
// `unsupported` 分出这里两类失败中的哪一类，调用方要用它们做完全不同的事
// （见 `support/CapabilityShortfall.h`）：
//
//   `true`   这张图 MLIR 这一侧收不了——某个 stage 的每一个模板都被三条判据里的某一条拒了。
//            legacy 收得了，应当回落给它。
//   `false`  调度结构坏了：没有子图 / plan / stage，或坐标越界。回落会把它藏起来。
//
// **缺省是 `false`**，与那个头文件里「没有显式标记的失败一律按坏了处置」同向：漏标只多报一次
// 失败，反过来会把真 bug 静默回落掉。标记本身不在这里发——这一层没有 MLIR context，由
// `RunMlirCodegen` 用它去发那条带标记的诊断，判据因此仍然只有一个。
bool BuildAllCandidateSelections(const ascir::FusedScheduledResult &fused_result,
                                 std::vector<PlanSelection> &selections, std::string *diagnostic = nullptr,
                                 bool *unsupported = nullptr);

// Remove templates that cannot be lowered by the MLIR implementation before
// narrowing the schedule. BuildAllCandidateSelections intentionally retains
// their original coordinates for ATT/runtime-key stability.
bool FilterSelectionsForLowering(const ascir::FusedScheduledResult &fused_result,
                                 std::vector<PlanSelection> &selections, std::string *diagnostic = nullptr);

// Copy only the selected plans/templates and renumber their coordinates from
// zero within each subgraph/stage, preserving selection order. Every subgraph
// must be covered; duplicate plan/template ids are invalid. On failure, the
// result is unchanged and narrowed_selections is empty.
bool NarrowFusedResultToSelections(const ascir::FusedScheduledResult &fused_result,
                                   const std::vector<PlanSelection> &selections,
                                   ascir::FusedScheduledResult &narrowed_result,
                                   std::vector<PlanSelection> &narrowed_selections, std::string *diagnostic = nullptr);

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_PLAN_SELECTOR_PLAN_SELECTION_H
