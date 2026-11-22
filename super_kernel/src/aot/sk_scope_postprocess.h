/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_scope_postprocess.h
 * \brief
 */

#ifndef __SK_SCOPE_POSTPROCESS_H__
#define __SK_SCOPE_POSTPROCESS_H__

#include "sk_graph.h"
#include "sk_scope_split.h"
#include <deque>
#include <memory>
#include <utility>
#include <vector>

// Read-only scheduling view; does not change nodes, graph links or event resources.
std::vector<SuperKernelBaseNode *> FilterCancelledNodes(const std::vector<SuperKernelBaseNode *> &nodes);

struct StreamPostPlan {
  bool needFrontWait = false;
  bool needBackBlock = false;
  uint64_t candidateNodeId = INVALID_TASK_ID;
};

class SuperKernelScopePostProcessor {
 public:
  SuperKernelScopePostProcessor() = default;
  SuperKernelScopePostProcessor(SuperKernelGraph &graph) : graph(graph) {}
  ~SuperKernelScopePostProcessor() = default;
  // Return true when this scope is processable by scheduler.
  // Return false when this scope is unprocessable and should be skipped by caller.
  bool PostProcess(SuperKernelScopeInfo &scopeInfo);
  // Read-only eligibility check; does not schedule tasks or allocate event resources.
  bool HasSufficientTaskSlots(const std::vector<ScopeStreamInfo> &scopeStreamInfos);

 private:
  bool ValidateScopeStreamNodes(const SuperKernelScopeInfo &scopeInfo);
  bool ApplyEventMemoryForFilteredNodes(std::vector<SuperKernelBaseNode *> &filteredNodes,
                                        std::vector<SuperKernelBaseNode *> &needUpdateNodes);
  bool CollectStreamBoundaryPlans(const std::vector<ScopeStreamInfo> &scopeStreamInfos,
                                  std::vector<StreamPostPlan> &plans, uint32_t &needFrontWaitCount);
  bool ProcessSubStreamSyncEvents(SuperKernelScopeInfo &scopeInfo, ScopeExtInfo &tempExtInfo,
                                  std::vector<StreamPostPlan> &plans, uint32_t mainStreamIdx,
                                  const std::vector<uint32_t> &subStreamOrder, uint32_t needFrontWaitCount);
  bool FinalizePostProcess(SuperKernelScopeInfo &scopeInfo, ScopeExtInfo &tempExtInfo,
                           std::vector<SuperKernelBaseNode *> &needUpdateNodes);

  SuperKernelGraph &graph;
};

#endif  // __SK_SCOPE_POSTPROCESS_H__
