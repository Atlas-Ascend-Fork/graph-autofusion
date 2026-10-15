/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_VERIFY_GRAPH_H__
#define __SK_VERIFY_GRAPH_H__

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "sk_scope_info.h"
#include "super_kernel.h"

class SuperKernelGraph;

class ScopeVerifyGraphBuilder {
 public:
  // Build internal graph, initial scopes, and node lookup map from scope verify input.
  aclError Build(const aclskScopeVerifyGraphInfo *verifyGraph, SuperKernelGraph &graph,
                 std::vector<SuperKernelScopeInfo> &initialScopes,
                 std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> &verifyNodesById);

 private:
  // Clear temporary stream and scope indexes before a new build.
  void ResetBuildState();

  // Resolve external streamId to internal graph stream index.
  aclError ResolveStream(SuperKernelGraph &graph, int64_t streamId, uint32_t &streamIdx);

  // Resolve external scopeId to an initial scope entry.
  SuperKernelScopeInfo &ResolveScope(int32_t scopeId, std::vector<SuperKernelScopeInfo> &initialScopes);

  // Validate, create, and append one scope verify node into graph and scope.
  aclError AppendNode(SuperKernelGraph &graph, aclskScopeVerifyNodeInfo &verifyNode,
                      std::vector<SuperKernelScopeInfo> &initialScopes,
                      std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> &verifyNodesById);

  std::unordered_map<int64_t, uint32_t> streamIdxById_;
  std::unordered_map<int32_t, size_t> scopeIdxById_;
  std::vector<uint64_t> preNodeIds_;
};

#endif  // __SK_VERIFY_GRAPH_H__
