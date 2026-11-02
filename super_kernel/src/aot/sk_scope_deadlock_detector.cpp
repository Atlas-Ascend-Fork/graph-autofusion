/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_scope_deadlock_detector.h"
#include "sk_log.h"
#include <algorithm>
#include <functional>

std::vector<const SuperKernelScopeInfo *> ScopeDeadlockDetector::CollectWaitScopeCandidates(
    const std::vector<SuperKernelScopeInfo> &scopes) const {
  std::vector<const SuperKernelScopeInfo *> candidates;
  for (const auto &scope : scopes) {
    const bool hasCheckableWait =
        std::any_of(scope.GetNodes().begin(), scope.GetNodes().end(), [this](const auto *node) {
          return node->GetNodeType() == SkNodeType::NODE_WAIT && !analysis_.ShouldBypassValueWaitDeadlock(*node);
        });
    if (scope.GetScopeCoreInfo().IsValid() && hasCheckableWait) {
      candidates.push_back(&scope);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto *lhs, const auto *rhs) { return lhs->GetScopeId() < rhs->GetScopeId(); });
  return candidates;
}

ScopeDeadlockDetector::ScopeConcurrentRanges ScopeDeadlockDetector::BuildScopeConcurrentRanges(
    const std::vector<SuperKernelScopeInfo> &scopes) const {
  ScopeConcurrentRanges ranges;
  for (const auto &scope : scopes) {
    const bool hasWait = std::any_of(scope.GetNodes().begin(), scope.GetNodes().end(),
                                     [](const auto *node) { return node->GetNodeType() == SkNodeType::NODE_WAIT; });
    if (scope.GetScopeCoreInfo().IsValid() && hasWait) {
      ranges.emplace(scope.GetScopeId(),
                     analysis_.CalculateScopeConcurrentRanges(scope.GetNodes(), scope.GetScopeId()));
    }
  }
  return ranges;
}

bool ScopeDeadlockDetector::CheckSelectedScopeResourceBranches(
    const std::vector<const SuperKernelScopeInfo *> &selected, const ScopeConcurrentRanges &ranges,
    uint64_t &riskWaitId, const SuperKernelScopeInfo *&riskScope) {
  if (selected.size() < 2) {
    return true;
  }
  std::unordered_set<uint16_t> residents;
  for (const auto *scope : selected) {
    residents.insert(scope->GetScopeId());
  }
  for (const auto *scope : selected) {
    const auto scopeId = scope->GetScopeId();
    const LockDetector::ResourceDemand rootDemand{scope->GetScopeCoreInfo().GetCubeNum(),
                                                  scope->GetScopeCoreInfo().GetVectorNum()};
    for (const auto *node : scope->GetNodes()) {
      if (node->GetNodeType() == SkNodeType::NODE_WAIT && !analysis_.ShouldBypassValueWaitDeadlock(*node) &&
          !analysis_.CheckWaitResourceBranches(*node, scopeId, rootDemand, scope->GetNodes(), residents, &ranges)) {
        riskWaitId = node->GetNodeId();
        riskScope = scope;
        SK_LOGI("[ParallelResource] Risk Wait %lu in scope %u; %zu resident scopes", riskWaitId, scopeId,
                selected.size());
        for (const auto *resident : selected) {
          SK_LOGI("[ParallelResource] Resident scope %u: cube=%u, vec=%u", resident->GetScopeId(),
                  resident->GetScopeCoreInfo().GetCubeNum(), resident->GetScopeCoreInfo().GetVectorNum());
        }
        return false;
      }
    }
  }
  return true;
}

const SuperKernelScopeInfo *ScopeDeadlockDetector::CheckParallelScopeResources(
    const std::vector<SuperKernelScopeInfo> &scopes, uint64_t &riskWaitId) {
  riskWaitId = INVALID_TASK_ID;
  const SuperKernelScopeInfo *riskScope = nullptr;
  analysis_.SetScopeInfos(scopes);
  const auto candidates = CollectWaitScopeCandidates(scopes);
  const auto ranges = BuildScopeConcurrentRanges(scopes);
  std::vector<const SuperKernelScopeInfo *> selected;
  LockDetector::ResourceDemand selectedDemand;
  std::function<bool(size_t)> check = [&](size_t index) {
    if (index == candidates.size()) {
      return CheckSelectedScopeResourceBranches(selected, ranges, riskWaitId, riskScope);
    }
    if (!check(index + 1)) {
      return false;
    }
    const auto *candidate = candidates[index];
    for (const auto *scope : selected) {
      if (!analysis_.CanRunTogether(scope->GetScopeId(), candidate->GetScopeId(), ranges)) {
        return true;
      }
    }
    const uint64_t candidateCube = selectedDemand.cube + candidate->GetScopeCoreInfo().GetCubeNum();
    const uint64_t candidateVec = selectedDemand.vec + candidate->GetScopeCoreInfo().GetVectorNum();
    if (!analysis_.HasEnoughDeviceCores(candidateCube, candidateVec, 0, 0)) {
      return true;
    }
    const auto previousDemand = selectedDemand;
    selectedDemand = {candidateCube, candidateVec};
    selected.push_back(candidate);
    const bool safe = check(index + 1);
    selected.pop_back();
    selectedDemand = previousDemand;
    return safe;
  };
  (void)check(0);
  return riskScope;
}
