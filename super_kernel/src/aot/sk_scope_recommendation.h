/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_SCOPE_RECOMMENDATION_H__
#define __SK_SCOPE_RECOMMENDATION_H__

#include <cstdint>
#include <limits>
#include <unordered_map>

class SuperKernelGraph;

class ScopeRecommendation {
 public:
  static constexpr uint64_t INVALID_CANDIDATE_ID = std::numeric_limits<uint64_t>::max();

  bool Analyze(const SuperKernelGraph &graph);
  uint64_t GetCandidateId(uint64_t nodeId) const {
    const auto iter = candidateByNode_.find(nodeId);
    return iter == candidateByNode_.end() ? INVALID_CANDIDATE_ID : iter->second;
  }

 private:
  std::unordered_map<uint64_t, uint64_t> candidateByNode_;
};

#endif  // __SK_SCOPE_RECOMMENDATION_H__
