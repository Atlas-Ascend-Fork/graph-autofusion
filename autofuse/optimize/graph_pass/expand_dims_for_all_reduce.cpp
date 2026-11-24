/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "expand_dims_for_all_reduce.h"

#include "ascir_utils.h"
#include "schedule_utils.h"
#include "util/mem_utils.h"

namespace optimize {

bool IsAllReduce(af::AscNode &node) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  std::vector<ascir::SizeExpr> src_strides;
  std::vector<ascir::SizeExpr> dst_strides = node.outputs[0].attr.strides;
  GE_CHK_STATUS_RET(ScheduleUtils::GetReduceInputStrides(node, src_strides), "Get loop strides failed.");
  GE_ASSERT_TRUE((src_strides.size() == node.outputs[0].attr.strides.size()),
                 "The output dim cnt [%zu] of reduce mismatch with input dim cnt [%zu].", dst_strides.size(),
                 src_strides.size());
  GE_ASSERT_TRUE((src_strides.size() == axes.size()),
                 "The input dim cnt [%zu] of reduce mismatch with reduce axes cnt [%zu].", src_strides.size(),
                 axes.size());
  std::vector<ascir::AxisId> reduce_axes;
  for (size_t i = 0UL; i < src_strides.size(); ++i) {
    if (src_strides[i] != dst_strides[i] && dst_strides[i] == 0) {
      reduce_axes.push_back(axes[i]);
    }
  }
  return reduce_axes.size() == axes.size();
}

Status ExpandDimsAtFirst(ascir::ImplGraph &owner_graph, const std::string &name, const af::Expression &size) {
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(owner_graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  if (graph_attr == nullptr) {
    GELOGE(af::FAILED, "Get or create graph attr failed for graph: %s", owner_graph.GetName().c_str());
    return af::FAILED;
  }
  GELOGD("before: axes = %s", ScheduleUtils::AxesToString(graph_attr->axis).c_str());
  const auto src_axes = graph_attr->axis;  // copy
  std::vector<af::AxisPtr> new_axes;
  std::shared_ptr<af::Axis> const_axis = af::MakeShared<af::Axis>();
  GE_CHECK_NOTNULL(const_axis, " create axis failed");
  const_axis->id = 0;
  const_axis->name = name;
  const_axis->type = af::Axis::kAxisTypeOriginal;
  const_axis->size = size;
  new_axes.push_back(std::move(const_axis));
  for (const auto &src_axis : src_axes) {
    std::shared_ptr<af::Axis> new_axis = af::MakeShared<af::Axis>();
    GE_CHECK_NOTNULL(new_axis, " create axis failed");
    new_axis->id = src_axis->id + 1;
    new_axis->name = src_axis->name;
    new_axis->type = src_axis->type;
    new_axis->size = src_axis->size;
    new_axes.push_back(std::move(new_axis));
  }
  graph_attr->axis = std::move(new_axes);
  GELOGD("after: axes = %s", ScheduleUtils::AxesToString(graph_attr->axis).c_str());
  return af::SUCCESS;
}

Status ExpandDimsForAllReducePass::RunPass(af::AscGraph &graph) {
  std::vector<ascir::AxisId> old_axis_ids;
  std::vector<ascir::AxisId> new_axis_ids;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->attr.api.compute_type == af::ComputeType::kComputeReduce) {
      if (!IsAllReduce(*node)) {
        continue;
      }
      GE_CHK_STATUS_RET(ExpandDimsAtFirst(graph, "axis_1d", af::ops::One), "Expand dims at first failed");
      old_axis_ids = node->attr.sched.axis;
      new_axis_ids.insert(new_axis_ids.end(), old_axis_ids.begin(), old_axis_ids.end());
      new_axis_ids.push_back(static_cast<int64_t>(new_axis_ids.size()));
      break;
    }
  }
  if (new_axis_ids.empty()) {
    return af::SUCCESS;
  }
  GELOGD("Expand dims for all reduce graph:%s", graph.GetName().c_str());
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsIOBuffer(node)) {
      continue;
    }
    auto cur_axis_ids = node->attr.sched.axis;
    GE_ASSERT_TRUE(!cur_axis_ids.empty() && cur_axis_ids == old_axis_ids,
                   "Expand dims for all reduce failed node:%s, Axis id mismatches with reduce, cannot be modified.",
                   node->GetName().c_str());
    node->attr.sched.axis = new_axis_ids;
    for (const auto output_attr : node->outputs()) {
      output_attr->attr.axis = new_axis_ids;
      GE_ASSERT_TRUE(output_attr->attr.strides.size() == old_axis_ids.size(),
                     "Expand dims for all reduce failed node:%s, Strides mismatch with reduce, cannot be modified.",
                     node->GetName().c_str());
      GE_ASSERT_TRUE(output_attr->attr.repeats.size() == old_axis_ids.size(),
                     "Expand dims for all reduce failed node:%s, Repeats mismatch with reduce, cannot be modified.",
                     node->GetName().c_str());
      if (output_attr->attr.strides[0UL] == 0) {
        output_attr->attr.strides.insert(output_attr->attr.strides.begin(), af::ops::One);
      } else {
        output_attr->attr.strides.insert(output_attr->attr.strides.begin(),
                                         af::sym::Mul(output_attr->attr.repeats[0UL], output_attr->attr.strides[0UL]));
      }
      output_attr->attr.repeats.insert(output_attr->attr.repeats.begin(), af::ops::One);
    }
  }
  return af::SUCCESS;
}
}  // namespace optimize
