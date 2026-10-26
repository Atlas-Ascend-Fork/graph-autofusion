/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "load_to_nddma_template.h"
#include "graph_utils.h"
#include "indirect_load_utils.h"
#include "common_utils.h"
#include "un_alignment_strategy.h"
#include "tensor_layout_utils.h"
#include "platform/common/base_alignment_strategy.h"

namespace optimize {
namespace semantic = ascgen_utils::template_semantic;

std::string LoadToNddmaTemplate::GenName(const std::string &general_case_name) {
  return general_case_name + "_load_to_nddma";
}

/**
 * 生成NDDMA模版，具体逻辑如下：
 * 1. 首先判断原图是否包含transpose节点，若包含则不能生成nddma模板
 * 2. 遍历图上所有节点，找到load节点时，判断是否为输出多引用，若不是则继续执行步骤2
 * 3. 判断load节点的尾轴是否做了transpose，若是则将load替换为nddma节点
 */
af::Status LoadToNddmaTemplate::Generate(const af::AscGraph &origin_graph,
                                         [[maybe_unused]] const af::AscGraph &based_case, af::AscGraph &new_case) {
  if (ScheduleUtils::HasComputeType(origin_graph, af::ComputeType::kComputeTranspose)) {
    GELOGD("No load_to_nddma template generated because origin_graph contains Transpose nodes.");
    return af::FAILED;
  }
  bool is_nddma_generated = false;
  for (const auto &node : new_case.GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (!af::ops::IsOps<af::ascir_op::Load>(node)) {
      continue;
    }
    if (ascgen_utils::indirect_load::GetTemplateBehavior(node).uses_direct_gm_pipeline) {
      continue;
    }
    ascgen_utils::DiscontinuityInfo info;
    GE_ASSERT_SUCCESS(ascgen_utils::TensorLayoutUtils::AnalyzeLoadDiscontinuity(node->outputs[0].attr, info),
                      "Failed to analyze discontinuity info for node:[%s].", node->GetNamePtr());
    bool need_align_at_repeat1 = info.has_multiple_discontinuities && info.is_tail_axis_discontinuous;
    if (IsLoadNeedAlign(node) || need_align_at_repeat1) {
      GE_ASSERT_SUCCESS(GenLoadToGenNddmaNode(node));
      GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticTransform(node, semantic::kLoad));
      is_nddma_generated = true;
    }
  }
  if (!is_nddma_generated) {
    GELOGD("No load_to_nddma template generated.");
    return af::FAILED;
  }
  GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(new_case));
  return af::SUCCESS;
}

bool LoadToNddmaTemplate::NeedDropBasedCase([[maybe_unused]] const af::AscGraph &origin_graph,
                                            [[maybe_unused]] const af::AscGraph &based_case,
                                            [[maybe_unused]] const af::AscGraph &new_case) {
  return true;
}

}  // namespace optimize
