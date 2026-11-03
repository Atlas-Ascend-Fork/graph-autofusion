/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelection.h"

#include <numeric>
#include <set>
#include <utility>

#include "bridge/BridgeDiagnostic.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace mlir::bridge {
namespace {

bool ValidatePlanStructure(const ascir::ScheduledResult &plan, size_t subgraph_id, size_t plan_id,
                           std::string *diagnostic) {
  const std::string location = "subgraph " + std::to_string(subgraph_id) + " plan " + std::to_string(plan_id);
  if (plan.schedule_groups.empty()) {
    SetDiagnostic(diagnostic, location + " contains no stage");
    return false;
  }
  for (const auto &stage : plan.schedule_groups) {
    if (stage.impl_graphs.empty()) {
      SetDiagnostic(diagnostic, location + " contains an empty stage");
      return false;
    }
  }
  return true;
}

bool HasValidVectorizedAxes(const af::AscTensorAttr &tensor) {
  if (tensor.axis.size() != tensor.repeats.size() ||
      tensor.vectorized_axis.size() != tensor.vectorized_strides.size()) {
    return false;
  }

  std::map<int64_t, af::Expression> repeats;
  for (size_t index = 0U; index < tensor.axis.size(); ++index) {
    repeats.emplace(tensor.axis[index], tensor.repeats[index]);
  }

  struct VectorAxis {
    af::Expression stride;
    af::Expression repeat;
  };
  std::vector<VectorAxis> axes;
  axes.reserve(tensor.vectorized_axis.size());
  for (size_t index = 0U; index < tensor.vectorized_axis.size(); ++index) {
    const int64_t axis = tensor.vectorized_axis[index];
    auto repeat = repeats.find(axis);
    if (repeat == repeats.end()) {
      return false;
    }
    if (af::SymbolicUtils::StaticCheckEq(repeat->second, af::sym::kSymbolOne) == af::TriBool::kTrue) {
      continue;
    }
    axes.push_back({tensor.vectorized_strides[index], repeat->second});
  }
  return true;
}

bool IsStaticOne(const af::Expression &expression) {
  return af::SymbolicUtils::StaticCheckEq(expression, af::sym::kSymbolOne) == af::TriBool::kTrue;
}

bool IsStaticZero(const af::Expression &expression) {
  return af::SymbolicUtils::StaticCheckEq(expression, af::sym::kSymbolZero) == af::TriBool::kTrue;
}

bool AxisIsVectorized(const af::AscTensorAttr &tensor, size_t axisIndex) {
  return std::find(tensor.vectorized_axis.begin(), tensor.vectorized_axis.end(), tensor.axis[axisIndex]) !=
         tensor.vectorized_axis.end();
}

bool IsScalarLikeInput(const af::Node &node) {
  const std::string type = node.GetType();
  return type == "Constant" || type == "IndexExpr" || type == "Scalar" || type == "ScalarData";
}

// The lowering-side shape predicate expressed on scheduler attributes. When a
// template needs inline broadcast, every differing repeat must be a size-one,
// zero-stride axis on a non-vectorized dimension. A non-broadcast outer stride
// may differ only when a later axis broadcasts, which changes the compact row
// pitch of the smaller tile.
bool BroadcastStridesAreCompatible(const af::AscTensorAttr &input, const af::AscTensorAttr &output,
                                   const std::vector<bool> &broadcastAxis) {
  auto hasLaterBroadcastAxis = [&broadcastAxis](size_t axis) {
    return std::any_of(broadcastAxis.begin() + static_cast<ptrdiff_t>(axis) + 1, broadcastAxis.end(),
                       [](bool value) { return value; });
  };
  for (size_t axis = 0U; axis < input.axis.size(); ++axis) {
    const bool sameRepeat =
        af::SymbolicUtils::StaticCheckEq(input.repeats[axis], output.repeats[axis]) == af::TriBool::kTrue;
    const bool sameStride =
        af::SymbolicUtils::StaticCheckEq(input.strides[axis], output.strides[axis]) == af::TriBool::kTrue;
    if (sameRepeat && sameStride) {
      continue;
    }
    if (broadcastAxis[axis]) {
      if (AxisIsVectorized(input, axis) || AxisIsVectorized(output, axis)) {
        return false;
      }
      continue;
    }
    if (!sameRepeat || !hasLaterBroadcastAxis(axis)) {
      return false;
    }

    af::Expression expectedStride = input.strides[axis];
    for (size_t broadcastAxisId = axis + 1U; broadcastAxisId < broadcastAxis.size(); ++broadcastAxisId) {
      if (!broadcastAxis[broadcastAxisId]) {
        continue;
      }
      expectedStride = expectedStride * output.repeats[broadcastAxisId];
    }
    if (af::SymbolicUtils::StaticCheckEq(output.strides[axis], expectedStride) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

bool InputBroadcastsToOutput(const af::AscTensorAttr &input, const af::AscTensorAttr &output) {
  if (input.axis != output.axis || input.repeats.size() != input.axis.size() ||
      input.strides.size() != input.axis.size() || output.repeats.size() != output.axis.size() ||
      output.strides.size() != output.axis.size()) {
    return false;
  }

  std::vector<bool> broadcastAxis(input.axis.size(), false);
  bool sawBroadcastAxis = false;
  bool sameView = true;
  for (size_t axis = 0U; axis < input.axis.size(); ++axis) {
    const bool sameRepeat =
        af::SymbolicUtils::StaticCheckEq(input.repeats[axis], output.repeats[axis]) == af::TriBool::kTrue;
    const bool sameStride =
        af::SymbolicUtils::StaticCheckEq(input.strides[axis], output.strides[axis]) == af::TriBool::kTrue;
    if (sameRepeat && sameStride) {
      continue;
    }
    sameView = false;
    broadcastAxis[axis] = (IsStaticOne(input.repeats[axis]) && IsStaticZero(input.strides[axis])) ||
                          (IsStaticOne(output.repeats[axis]) && IsStaticZero(output.strides[axis]));
    sawBroadcastAxis = sawBroadcastAxis || broadcastAxis[axis];
  }
  if (sameView) {
    return true;
  }
  if (!sawBroadcastAxis) {
    return false;
  }

  return BroadcastStridesAreCompatible(input, output, broadcastAxis);
}

struct QueueResourceKey {
  int64_t queueId;
  int64_t reuseId;

  bool operator<(const QueueResourceKey &other) const {
    return std::tie(queueId, reuseId) < std::tie(other.queueId, other.reuseId);
  }
};

struct TensorViewFacts {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;

  bool operator==(const TensorViewFacts &other) const {
    return axis == other.axis && repeats == other.repeats && strides == other.strides;
  }
};

TensorViewFacts ViewFacts(const af::AscTensorAttr &tensor) {
  return {tensor.axis, tensor.repeats, tensor.strides};
}

// A queue slot has one materialized tile. The ScheduleToAfir resource verifier
// therefore rejects a template that puts two different views on the same queue
// and reuse identity. Every queue view is defined by a producer output in this
// schedule representation, so collecting node outputs covers all queue views.
// Filter that template here so implementation selection does not emit a candidate
// diagnostic while another template succeeds.
bool TemplateSupportsMlirQueueResources(const af::AscGraph &graph) {
  std::map<QueueResourceKey, std::vector<TensorViewFacts>> views;
  for (const auto &node : graph.GetAllNodes()) {
    for (const auto &output : node->outputs()) {
      const auto &attr = output->attr;
      if (attr.mem.alloc_type != af::AllocType::kAllocTypeQueue) {
        continue;
      }
      views[{attr.que.id, attr.mem.reuse_id}].push_back(ViewFacts(attr));
    }
  }

  for (const auto &[key, keyViews] : views) {
    (void)key;
    for (size_t first = 0U; first < keyViews.size(); ++first) {
      for (size_t second = first + 1U; second < keyViews.size(); ++second) {
        if (!(keyViews[first] == keyViews[second])) {
          return false;
        }
      }
    }
  }
  return true;
}

bool TemplateSupportsMlirElementwiseBroadcast(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->attr.api.compute_type != af::ComputeType::kComputeElewise || node->outputs().empty()) {
      continue;
    }
    const af::AscTensorAttr &output = node->outputs()[0]->attr;
    for (const auto &input : node->inputs()) {
      const af::Node *inputNode = input == nullptr ? nullptr : input->anchor.GetOwnerNodeBarePtr();
      if (input == nullptr || input->attr.axis.empty() || (inputNode != nullptr && IsScalarLikeInput(*inputNode))) {
        continue;
      }
      if (!InputBroadcastsToOutput(input->attr, output)) {
        return false;
      }
    }
  }
  return true;
}

// DataMovement lowers multi-segment local views through TransferPlan's outer
// loops, emitting one DataCopyPadExtend per segment. Only malformed descriptor
// metadata is rejected here; the reg-base-only NDDMA path has its own AFIR
// capability check.
bool TemplateSupportsMlirDataMovement(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Load") {
      if (node->outputs().empty() || !HasValidVectorizedAxes(node->outputs()[0]->attr)) {
        return false;
      }
    } else if (node->GetType() == "Store") {
      if (node->inputs().empty() || !HasValidVectorizedAxes(node->inputs()[0]->attr)) {
        return false;
      }
    }
  }
  return true;
}

// 三条模板判据各自拒掉了多少个模板。
//
// **拒收归属必须由代码报出来，不能按形状去读。** 这一层今天只报「没有 MLIR 支持的模板」，
// 不说是哪一条判据不满足；按用例形状反推过一次，5 条里猜错了（见 2026-10-21 的归因）。
//
// 三条都要算，不短路：短路只省掉拒收路径上的两次图遍历（与后面的下降相比可忽略），换来的
// 是「一个模板同时违反几条」这件事看得见——按第一条报会把并列的判据说成唯一判据。
struct TemplateRejection {
  size_t dataMovement = 0U;
  size_t elementwiseBroadcast = 0U;
  size_t queueResources = 0U;
  size_t templates = 0U;

  // **首因是「拒掉了全部模板」的那一条，不是按某个顺序排在前面的那一条。** 只拒掉其中几个
  // 的判据不是卡点——换一个模板它就不成立，真正挡住这个 stage 的是那条把所有模板都拒了的。
  //
  // 这条规则是量出来的：`indirect_load_both_transpose_simd_fallback` 等 3 格的 stage 有 2 个
  // 模板，queue-resources 把 2 个全拒了，而 elementwise-broadcast 只拒了 1 个。按「data-movement
  // → elementwise-broadcast → queue-resources」的顺序报第一条非零的，会把这 3 格的首因答成
  // elementwise-broadcast / data-movement（2026-10-26 实测，与 2026-10-20 的逐格表差 3 格，
  // 差的就是这个规则）。
  std::string Str() const {
    const auto join = [](const std::vector<std::string> &names) {
      std::string text;
      for (const auto &name : names) {
        text += (text.empty() ? "" : ", ") + name;
      }
      return text;
    };
    std::vector<std::string> binding;
    std::vector<std::string> partial;
    const auto bucket = [&](size_t count, const char *name) {
      if (count == 0U) {
        return;
      }
      (count == templates ? binding : partial).push_back(std::string(name) + " " + std::to_string(count));
    };
    bucket(dataMovement, "data-movement");
    bucket(elementwiseBroadcast, "elementwise-broadcast");
    bucket(queueResources, "queue-resources");
    // **「没有哪一条单独拒掉全部模板」是会发生的**：两个模板各被不同的一条拒掉，覆盖是一个
    // 组合而不是单条判据。`af-capability-probe` 的 T3 `[1,8] -> [128,8]` 与
    // `[1,1,8] -> [4,4,8]` 就是这样（2026-10-26 实测）。这时候报不出单一首因，只能说清是哪
    // 几条合起来挡住的——**不许退回挑一条**，那等于把组合谎称成单条。
    if (binding.empty()) {
      return std::to_string(templates) + " template(s), no single predicate rejects all; counts: " + join(partial);
    }
    std::string text = std::to_string(templates) + " template(s), all rejected by: " + join(binding);
    if (!partial.empty()) {
      text += "; partially: " + join(partial);
    }
    return text;
  }
};

bool TemplateSupportsMlir(const af::AscGraph &graph, TemplateRejection &rejection) {
  bool supported = true;
  if (!TemplateSupportsMlirDataMovement(graph)) {
    ++rejection.dataMovement;
    supported = false;
  }
  if (!TemplateSupportsMlirElementwiseBroadcast(graph)) {
    ++rejection.elementwiseBroadcast;
    supported = false;
  }
  if (!TemplateSupportsMlirQueueResources(graph)) {
    ++rejection.queueResources;
    supported = false;
  }
  if (!supported) {
    ++rejection.templates;
  }
  return supported;
}

std::vector<size_t> MakeContiguousIds(size_t count) {
  std::vector<size_t> ids(count);
  std::iota(ids.begin(), ids.end(), size_t{0U});
  return ids;
}

}  // namespace

bool ValidatePlanSelection(const ascir::FusedScheduledResult &fused_result, const PlanSelection &selection,
                           std::string *diagnostic) {
  if (selection.subgraph_id >= fused_result.node_idx_to_scheduled_results.size()) {
    SetDiagnostic(diagnostic, "subgraph_id is out of range");
    return false;
  }
  const auto &plans = fused_result.node_idx_to_scheduled_results[selection.subgraph_id];
  if (selection.plan_id >= plans.size()) {
    SetDiagnostic(diagnostic, "plan_id is out of range");
    return false;
  }
  const auto &plan = plans[selection.plan_id];
  if (!ValidatePlanStructure(plan, selection.subgraph_id, selection.plan_id, diagnostic)) {
    return false;
  }
  const auto &stages = plan.schedule_groups;
  if (selection.template_ids_per_stage.size() != stages.size()) {
    SetDiagnostic(diagnostic, "template_ids_per_stage does not cover every stage");
    return false;
  }
  for (size_t stage_id = 0U; stage_id < stages.size(); ++stage_id) {
    const auto &templates = selection.template_ids_per_stage[stage_id];
    const std::string location = " for stage " + std::to_string(stage_id);
    if (templates.empty()) {
      SetDiagnostic(diagnostic, "template_ids is empty" + location);
      return false;
    }
    std::set<size_t> seen;
    for (size_t template_id : templates) {
      if (template_id >= stages[stage_id].impl_graphs.size()) {
        SetDiagnostic(diagnostic, "template_id is out of range" + location);
        return false;
      }
      if (!seen.insert(template_id).second) {
        SetDiagnostic(diagnostic, "duplicate template_id" + location);
        return false;
      }
    }
  }
  return true;
}

bool BuildAllCandidateSelections(const ascir::FusedScheduledResult &fused_result,
                                 std::vector<PlanSelection> &selections, std::string *diagnostic, bool *unsupported) {
  selections.clear();
  const auto &subgraphs = fused_result.node_idx_to_scheduled_results;
  if (subgraphs.empty()) {
    SetDiagnostic(diagnostic, "fused schedule contains no candidate plan");
    return false;
  }
  std::vector<PlanSelection> candidates;
  std::vector<bool> subgraphHasCandidate(subgraphs.size(), false);
  // 每个子图第一次「某个 stage 一个模板都不剩」时的拒收归属，拿来写诊断。只留第一次：后面
  // 的 plan 再拒一次不改变结论，而多条拼在一起读不出首因。
  std::vector<std::string> subgraphRejection(subgraphs.size());
  for (size_t subgraph_id = 0U; subgraph_id < subgraphs.size(); ++subgraph_id) {
    const auto &plans = subgraphs[subgraph_id];
    if (plans.empty()) {
      SetDiagnostic(diagnostic, "subgraph " + std::to_string(subgraph_id) + " contains no candidate plan");
      return false;
    }
    for (size_t plan_id = 0U; plan_id < plans.size(); ++plan_id) {
      const auto &plan = plans[plan_id];
      if (!ValidatePlanStructure(plan, subgraph_id, plan_id, diagnostic)) {
        return false;
      }
      PlanSelection candidate{subgraph_id, plan_id, {}};
      candidate.template_ids_per_stage.reserve(plan.schedule_groups.size());
      bool planIsSupported = true;
      for (size_t stage_id = 0U; stage_id < plan.schedule_groups.size(); ++stage_id) {
        const auto &stage = plan.schedule_groups[stage_id];
        TemplateRejection rejection;
        std::vector<size_t> supportedTemplateIds;
        supportedTemplateIds.reserve(stage.impl_graphs.size());
        for (size_t templateId = 0U; templateId < stage.impl_graphs.size(); ++templateId) {
          if (TemplateSupportsMlir(stage.impl_graphs[templateId], rejection)) {
            supportedTemplateIds.push_back(templateId);
          }
        }
        // Keep every original template id in the selection. The runtime
        // tiling key is defined by scheduler order, so removing unsupported
        // templates would renumber later templates and change the ABI.
        std::vector<size_t> templateIds(stage.impl_graphs.size());
        std::iota(templateIds.begin(), templateIds.end(), size_t{0U});
        candidate.template_ids_per_stage.push_back(std::move(templateIds));
        if (supportedTemplateIds.empty()) {
          planIsSupported = false;
          if (subgraphRejection[subgraph_id].empty()) {
            subgraphRejection[subgraph_id] =
                "plan " + std::to_string(plan_id) + " stage " + std::to_string(stage_id) + ": " + rejection.Str();
          }
        }
      }
      if (planIsSupported) {
        subgraphHasCandidate[subgraph_id] = true;
      }
      candidates.push_back(std::move(candidate));
    }
  }
  for (size_t subgraph_id = 0U; subgraph_id < subgraphHasCandidate.size(); ++subgraph_id) {
    if (!subgraphHasCandidate[subgraph_id]) {
      SetDiagnostic(diagnostic, "subgraph " + std::to_string(subgraph_id) +
                                    " contains no plan with an MLIR-supported template (" +
                                    subgraphRejection[subgraph_id] + ")");
      SetUnsupported(unsupported);
      return false;
    }
  }
  if (candidates.empty()) {
    SetDiagnostic(diagnostic, "fused schedule contains no MLIR-supported candidate plan");
    SetUnsupported(unsupported);
    return false;
  }
  selections = std::move(candidates);
  return true;
}

bool FilterSelectionsForLowering(const ascir::FusedScheduledResult &fused_result,
                                 std::vector<PlanSelection> &selections, std::string *diagnostic) {
  std::vector<PlanSelection> filtered;
  filtered.reserve(selections.size());
  std::vector<bool> covered(fused_result.node_idx_to_scheduled_results.size(), false);
  for (const auto &selection : selections) {
    if (!ValidatePlanSelection(fused_result, selection, diagnostic)) {
      return false;
    }
    const auto &plan = fused_result.node_idx_to_scheduled_results[selection.subgraph_id][selection.plan_id];
    PlanSelection lowered{selection.subgraph_id, selection.plan_id, {}};
    bool supported = true;
    for (size_t stage_id = 0U; stage_id < plan.schedule_groups.size(); ++stage_id) {
      const auto &stage = plan.schedule_groups[stage_id];
      std::vector<size_t> templateIds;
      for (size_t templateId : selection.template_ids_per_stage[stage_id]) {
        TemplateRejection rejection;
        if (TemplateSupportsMlir(stage.impl_graphs[templateId], rejection)) {
          templateIds.push_back(templateId);
        }
      }
      if (templateIds.empty()) {
        supported = false;
        break;
      }
      lowered.template_ids_per_stage.push_back(std::move(templateIds));
    }
    if (supported) {
      covered[selection.subgraph_id] = true;
      filtered.push_back(std::move(lowered));
    }
  }
  for (size_t subgraphId = 0U; subgraphId < covered.size(); ++subgraphId) {
    if (!covered[subgraphId]) {
      SetDiagnostic(diagnostic, "subgraph " + std::to_string(subgraphId) + " has no lowerable MLIR candidate");
      return false;
    }
  }
  selections = std::move(filtered);
  return true;
}

bool NarrowFusedResultToSelections(const ascir::FusedScheduledResult &fused_result,
                                   const std::vector<PlanSelection> &selections,
                                   ascir::FusedScheduledResult &narrowed_result,
                                   std::vector<PlanSelection> &narrowed_selections, std::string *diagnostic) {
  narrowed_selections.clear();
  const auto &subgraphs = fused_result.node_idx_to_scheduled_results;
  if (subgraphs.empty()) {
    SetDiagnostic(diagnostic, "fused schedule contains no candidate plan");
    return false;
  }
  std::vector<std::set<size_t>> selected_plans(subgraphs.size());
  for (const auto &selection : selections) {
    if (!ValidatePlanSelection(fused_result, selection, diagnostic)) {
      return false;
    }
    if (!selected_plans[selection.subgraph_id].insert(selection.plan_id).second) {
      SetDiagnostic(diagnostic, "duplicate plan_id for subgraph " + std::to_string(selection.subgraph_id));
      return false;
    }
  }
  for (size_t subgraph_id = 0U; subgraph_id < subgraphs.size(); ++subgraph_id) {
    if (selected_plans[subgraph_id].empty()) {
      SetDiagnostic(diagnostic, "subgraph " + std::to_string(subgraph_id) + " has no selected plan");
      return false;
    }
  }

  // Publish both outputs only after validation and construction are complete.
  // The source graphs and the fusion's other metadata are preserved by copying.
  ascir::FusedScheduledResult result = fused_result;
  for (auto &plans : result.node_idx_to_scheduled_results) {
    plans.clear();
  }
  std::vector<PlanSelection> remapped;
  remapped.reserve(selections.size());
  for (const auto &selection : selections) {
    auto plan = subgraphs[selection.subgraph_id][selection.plan_id];
    auto &plans = result.node_idx_to_scheduled_results[selection.subgraph_id];
    PlanSelection new_selection{selection.subgraph_id, plans.size(), {}};
    new_selection.template_ids_per_stage.reserve(plan.schedule_groups.size());
    for (size_t stage_id = 0U; stage_id < plan.schedule_groups.size(); ++stage_id) {
      auto &stage = plan.schedule_groups[stage_id];
      const auto &templates = selection.template_ids_per_stage[stage_id];
      std::vector<af::AscGraph> graphs;
      graphs.reserve(templates.size());
      for (size_t template_id : templates) {
        graphs.push_back(stage.impl_graphs[template_id]);
      }
      stage.impl_graphs = std::move(graphs);
      new_selection.template_ids_per_stage.push_back(MakeContiguousIds(templates.size()));
    }
    plans.push_back(std::move(plan));
    remapped.push_back(std::move(new_selection));
  }
  narrowed_result = std::move(result);
  narrowed_selections = std::move(remapped);
  return true;
}

}  // namespace mlir::bridge
