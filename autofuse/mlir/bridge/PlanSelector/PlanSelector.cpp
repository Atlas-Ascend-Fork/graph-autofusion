/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/PlanSelector/PlanSelector.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include "bridge/BridgeDiagnostic.h"
#include "bridge/PlanSelector/AttCostEvaluator.h"
#include "gen_model_info/gen_model_info.h"

namespace mlir::bridge {
namespace {

bool ValidateShapeSymbols(const ConcreteShape &shape, std::string *diagnostic) {
  for (const auto &[name, _] : shape) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name.front())) || name.front() == '_') ||
        !std::all_of(name.begin() + 1, name.end(), [](unsigned char ch) { return std::isalnum(ch) || ch == '_'; })) {
      SetDiagnostic(diagnostic, "static plan selection received an invalid shape symbol");
      return false;
    }
  }
  return true;
}

// BuildAllCandidateSelections has already checked that every subgraph has a
// plan and every plan/stage is nonempty, and enumerated each plan exactly once.
bool HasUniqueCandidatePerSubgraph(const std::vector<PlanSelection> &candidates, size_t subgraph_count) {
  if (candidates.size() != subgraph_count) {
    return false;
  }
  for (const auto &candidate : candidates) {
    for (const auto &templates : candidate.template_ids_per_stage) {
      if (templates.size() != 1U) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

bool SelectDynamicCandidates(const ascir::FusedScheduledResult &fused_result, std::vector<PlanSelection> &selections,
                             std::string *diagnostic, bool *unsupported) {
  return BuildAllCandidateSelections(fused_result, selections, diagnostic, unsupported);
}

bool SelectStaticWinner(const ascir::FusedScheduledResult &fused_result, const ConcreteShape &shape,
                        const AttOptions &options, std::vector<PlanSelection> &selections, std::string *diagnostic,
                        bool *unsupported) {
  selections.clear();
  if (!ValidateShapeSymbols(shape, diagnostic)) {
    return false;
  }
  std::vector<PlanSelection> candidates;
  if (!BuildAllCandidateSelections(fused_result, candidates, diagnostic, unsupported)) {
    return false;
  }
  if (HasUniqueCandidatePerSubgraph(candidates, fused_result.node_idx_to_scheduled_results.size())) {
    selections = std::move(candidates);
    return true;
  }

  // **ATT 代价模型答不出来是能力边界，不是坏了。** 它的 perf 公式按 dtype 逐张表查，表里
  // 没有那一位就整图生成不出代价式（2026-10-26：`kBrcbRepeatMap` 没有 bfloat16，于是 bf16 的
  // 分段广播在这里折返，探针 reg 侧 68 格）。那是 legacy 自己的代价模型的供给范围，不是 MLIR
  // 这一侧少做了什么——legacy 收不收这类图由它自己决定，MLIR 该做的是答「不支持」并回落，
  // 而不是替它把表补齐（补了会连 legacy 的 plan 选择一起改掉）。
  //
  // **这与 `AttSchemaAdapter.cpp` 的字段拼法不符那一支不同**，那一支是缺陷：两侧对同一个
  // tiling 字段的拼法对不上，标成能力不足就把它静默回落掉了（F-4 的判断，reg 侧 16 格）。
  // 判据是「ATT 答不出来」与「两侧对同一件事的编码不一致」之分，不是「失败发生在 ATT 里」。
  att::FusedParsedScheduleResult models;
  if (att::GetModelInfoMap(fused_result, options, models) != af::SUCCESS) {
    SetDiagnostic(diagnostic, "ATT model information generation failed");
    SetUnsupported(unsupported);
    return false;
  }
  if (detail::SelectWinnersFromModels(fused_result, models, shape, selections, diagnostic)) {
    return true;
  }

  // 代价模型代不出常数时**保留全部候选，交给生成的 tiling 在运行期选**，而不是报失败。
  //
  // 静态形状也代不出常数是常态，不是异常：ATT 的代价式里除了 shape 符号，还有它自己的
  // 求解量——分核数（`block_dim`）与逐轴的切分尺寸，都要等 tiling 解完才有值，而这一步
  // 跑在 tiling 之前。广播图是第一个把它暴露出来的（`AttCostEvaluatorTest` 喂的全是合成的
  // 常量 `ModelInfo`，真实图上这条路一次都没跑通过）。
  //
  // **回落到「全留」不是盲挑一个**：`PlanToAfir` 本来就按 `template_counts_per_stage` 建
  // 多模板的 AFIR，ATT 随后为每个模板各生成一份 tiling 与 score func，由生成的 host tiling
  // 按实际 shape 比分——legacy 就是这么选的（它压根不在编译期选模板）。挑第一个才是盲挑：
  // 那等于拿输入顺序冒充性能结论。
  //
  // 代价是 kernel 里多一份模板的代码与一次运行期比分；换来的是「代不出常数」从整图失败
  // 变成能走 MLIR。诊断不清掉——它记的是"为什么没在编译期定下来"，调用方不读它。
  //
  // **这里返回 false 不会回落 legacy。** 回落只发生在「能力不足」那一类失败上，判据是诊断
  // 上的标记（`support/CapabilityShortfall.h`）；本函数剩下的两条失败路径（shape 符号非法、
  // 候选枚举的结构性失败）都是坏了，`RunMlirCodegen` 一律 `kFailed`。ATT 模型生成失败那一条
  // 置了 `unsupported`，理由见上面那段。
  selections = std::move(candidates);
  return true;
}

}  // namespace mlir::bridge
