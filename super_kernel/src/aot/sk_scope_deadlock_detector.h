/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_SCOPE_DEADLOCK_DETECTOR_H__
#define __SK_SCOPE_DEADLOCK_DETECTOR_H__

#include "sk_lock_detector.h"

class ScopeDeadlockDetector {
 public:
  ScopeDeadlockDetector(SuperKernelGraph &graph, const SuperKernelOptionsManager &opts)
      : concurrentRanges_(graph, scopeInfos_),
        waitResources_(graph, &opts, scopeInfos_, concurrentRanges_, LockDetector::GetDeviceCubeNum(),
                       LockDetector::GetDeviceVecNum()) {
    LockDetector::GetDeviceCores();
  }

  // 返回风险Wait所属scope，并通过riskWaitId输出对应节点；无风险时返回nullptr。
  const SuperKernelScopeInfo *CheckParallelScopeResources(const std::vector<SuperKernelScopeInfo> &scopes,
                                                          uint64_t &riskWaitId);

 private:
  using ScopeConcurrentRanges = ScopeConcurrentRangeAnalyzer::ScopeConcurrentRanges;

  std::vector<const SuperKernelScopeInfo *> CollectWaitScopeCandidates(
      const std::vector<SuperKernelScopeInfo> &scopes) const;
  ScopeConcurrentRanges BuildScopeConcurrentRanges(const std::vector<SuperKernelScopeInfo> &scopes) const;
  bool CheckSelectedScopeResourceBranches(const std::vector<const SuperKernelScopeInfo *> &selected,
                                          const ScopeConcurrentRanges &ranges, uint64_t &riskWaitId,
                                          const SuperKernelScopeInfo *&riskScope);

  std::unordered_map<uint16_t, ScopeRuntimeInfo> scopeInfos_;
  ScopeConcurrentRangeAnalyzer concurrentRanges_;
  WaitResourceDependencyAnalyzer waitResources_;
};

#endif
