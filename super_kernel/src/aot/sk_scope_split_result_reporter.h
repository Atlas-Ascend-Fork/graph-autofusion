/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file sk_scope_split_result_reporter.h
 * \brief Records scope split results and restores break information after re-split
 */

#ifndef __SK_SCOPE_SPLIT_RESULT_REPORTER_H__
#define __SK_SCOPE_SPLIT_RESULT_REPORTER_H__

#include <bitset>
#include <vector>
#include "sk_scope_info.h"

class ScopeSplitResultReporter {
 public:
  bool HasSameScopeStructure(const SuperKernelScopeInfo &sourceScope, const SuperKernelScopeInfo &targetScope) const;

  void ReportNewBreak(SuperKernelScopeInfo &scope, ScopeBreakInfo breakInfo) const;
  void ReportInheritedBreak(SuperKernelScopeInfo &scope, const ScopeBreakInfo &sourceBreakInfo) const;

  void CaptureResplitScopes(const std::vector<SuperKernelScopeInfo> &scopes);
  void RestoreResplitBreakInfos(std::vector<SuperKernelScopeInfo> &scopes) const;
  void Reset();

 private:
  struct ScopeSnapshot {
    std::vector<uint64_t> kernelNodeIds;
    std::vector<uint64_t> defaultNodeIds;
    std::bitset<MAX_SCOPE_NUM> scopeBitFlags;
    ScopeBreakInfo breakInfo;
  };

  static std::vector<uint64_t> GetNodeIds(const SuperKernelScopeInfo &scope, SkNodeType nodeType, bool skipScopeNode);
  static ScopeSnapshot MakeSnapshot(const SuperKernelScopeInfo &scope);
  static bool HasSameScopeStructure(const ScopeSnapshot &snapshot, const SuperKernelScopeInfo &scope);

  std::vector<ScopeSnapshot> resplitScopeSnapshots_;
};

#endif  // __SK_SCOPE_SPLIT_RESULT_REPORTER_H__
