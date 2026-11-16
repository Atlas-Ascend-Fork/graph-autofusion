/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_H_
#define OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_H_

#include "ascir.h"
#include "tiling_group.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "schedule.h"
#include "autoschedule_defs.h"
#include "ascgen_log.h"
#include "optimize.h"

namespace optimize::autoschedule {
class AutoSchedule {
 public:
  AutoSchedule() = delete;
  explicit AutoSchedule(ascir::ImplGraph &graph, std::vector<AutoScheduleOutput> &schd_outputs,
                        bool is_reduce_first_stage = false,
                        optimize::ReduceTemplateType reduce_template = optimize::ReduceTemplateType::kDefault,
                        ascir::CubeTemplateType cube_template = ascir::CubeTemplateType::kDefault,
                        bool enable_workspace_alignment = false)
      : graph_(graph),
        schd_outputs_(schd_outputs),
        is_reduce_first_stage_(is_reduce_first_stage),
        reduce_template_(reduce_template),
        cube_template_(cube_template),
        enable_workspace_alignment_(enable_workspace_alignment) {};

  Status DoAutoSchedule();

 private:
  // 生成所有可能的TilingCase列表
  Status PrepareTilingCases(std::vector<TilingCase> &tiling_cases);

  // 针对每个TilingCase生成通用切分模版
  Status ProcessOneTilingCase(TilingCase &tiling_case, size_t index, bool is_last_axis_reduce,
                              bool is_reduce_full_load) const;

  // 生成UBFuse模板（用于Cube模板为kUBFuse的情况）
  void GenUBFuseTemplates() const;
  void GenTilingCase(std::vector<TilingCase> &tiling_cases);
  Status PruneTilingCase(std::vector<TilingCase> &tiling_cases) const;
  Status SelectLoopAxis(ascir::ImplGraph &impl_graph, bool is_reduce_fullload) const;

  ascir::ImplGraph &graph_;
  std::vector<AutoScheduleOutput> &schd_outputs_;
  AxisGroup axes_group_;
  bool is_reduce_first_stage_;
  optimize::ReduceTemplateType reduce_template_;
  ascir::CubeTemplateType cube_template_;
  bool enable_workspace_alignment_;
};
}  // namespace optimize::autoschedule

#endif
