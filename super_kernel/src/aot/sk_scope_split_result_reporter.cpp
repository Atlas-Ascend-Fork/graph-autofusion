/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_scope_split_result_reporter.h"
#include <algorithm>

std::vector<uint64_t> ScopeSplitResultReporter::GetNodeIds(const SuperKernelScopeInfo &scope, SkNodeType nodeType,
                                                           bool skipScopeNode) {
  std::vector<uint64_t> nodeIds;
  for (const auto *node : scope.GetNodes()) {
    if (node != nullptr && node->GetNodeType() == nodeType && (!skipScopeNode || !node->IsScopeNode())) {
      nodeIds.push_back(node->GetNodeId());
    }
  }
  std::sort(nodeIds.begin(), nodeIds.end());
  return nodeIds;
}

ScopeSplitResultReporter::ScopeSnapshot ScopeSplitResultReporter::MakeSnapshot(const SuperKernelScopeInfo &scope) {
  return {GetNodeIds(scope, SkNodeType::NODE_KERNEL, true), GetNodeIds(scope, SkNodeType::NODE_DEFAULT, false),
          scope.GetScopeBitFlags(), scope.GetBreakInfo()};
}

bool ScopeSplitResultReporter::HasSameScopeStructure(const ScopeSnapshot &snapshot, const SuperKernelScopeInfo &scope) {
  return snapshot.kernelNodeIds == GetNodeIds(scope, SkNodeType::NODE_KERNEL, true) &&
         snapshot.defaultNodeIds == GetNodeIds(scope, SkNodeType::NODE_DEFAULT, false) &&
         snapshot.scopeBitFlags == scope.GetScopeBitFlags();
}

bool ScopeSplitResultReporter::HasSameScopeStructure(const SuperKernelScopeInfo &sourceScope,
                                                     const SuperKernelScopeInfo &targetScope) const {
  return HasSameScopeStructure(MakeSnapshot(sourceScope), targetScope);
}

void ScopeSplitResultReporter::ReportNewBreak(SuperKernelScopeInfo &scope, ScopeBreakInfo breakInfo) const {
  breakInfo.SetParentScopeId(INVALID_SCOPE_ID);
  scope.SetBreakInfo(std::move(breakInfo));
}

void ScopeSplitResultReporter::ReportInheritedBreak(SuperKernelScopeInfo &scope,
                                                    const ScopeBreakInfo &sourceBreakInfo) const {
  scope.SetBreakInfo(sourceBreakInfo);
}

void ScopeSplitResultReporter::CaptureResplitScopes(const std::vector<SuperKernelScopeInfo> &scopes) {
  resplitScopeSnapshots_.clear();
  for (const auto &scope : scopes) {
    if (scope.GetBreakInfo().GetReason() != ScopeBreakReason::NONE) {
      resplitScopeSnapshots_.push_back(MakeSnapshot(scope));
    }
  }
}

void ScopeSplitResultReporter::RestoreResplitBreakInfos(std::vector<SuperKernelScopeInfo> &scopes) const {
  for (auto &scope : scopes) {
    const auto snapshotIt =
        std::find_if(resplitScopeSnapshots_.begin(), resplitScopeSnapshots_.end(),
                     [&scope](const ScopeSnapshot &snapshot) { return HasSameScopeStructure(snapshot, scope); });
    if (snapshotIt != resplitScopeSnapshots_.end()) {
      ReportInheritedBreak(scope, snapshotIt->breakInfo);
    }
  }
}

void ScopeSplitResultReporter::Reset() {
  resplitScopeSnapshots_.clear();
}
