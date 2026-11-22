/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_queue_type_resolver.h"

#include <algorithm>
#include <unordered_map>

#include "sk_graph.h"
#include "sk_log.h"
#include "sk_node.h"

namespace {
constexpr int kMaxKernelSearchHops = 100;

enum class SearchDirection : uint8_t { PREV, NEXT };

SkQueueType ToQueueType(SkKernelType kernelType) {
  switch (kernelType) {
    case SkKernelType::AIC_ONLY:
    case SkKernelType::MIX_AIC_1_0:
      return SkQueueType::AIC;
    case SkKernelType::AIV_ONLY:
    case SkKernelType::MIX_AIV_1_0:
      return SkQueueType::AIV;
    case SkKernelType::MIX_AIC_1_1:
      return SkQueueType::MIX_1_1;
    case SkKernelType::MIX_AIC_1_2:
      return SkQueueType::MIX_1_2;
    default:
      SK_LOGW("Unsupported kernel type %s for super kernel; using AIC as fallback", to_string(kernelType));
      return SkQueueType::AIC;
  }
}

bool UsesAic(SkQueueType type) {
  return type == SkQueueType::AIC || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

bool UsesAiv(SkQueueType type) {
  return type == SkQueueType::AIV || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

SkQueueType ToEventQueueType(SkQueueType queueType, bool aicAvailable, bool aivAvailable) {
  SkQueueType result = queueType == SkQueueType::AIC ? SkQueueType::AIC : SkQueueType::AIV;
  if (result == SkQueueType::AIC && !aicAvailable) {
    return SkQueueType::AIV;
  }
  if (result == SkQueueType::AIV && !aivAvailable) {
    return SkQueueType::AIC;
  }
  return result;
}

const SuperKernelBaseNode *FindKernelNodeInDirection(uint64_t startNodeId, const SuperKernelGraph &graph,
                                                     SearchDirection direction,
                                                     std::unordered_map<uint64_t, const SuperKernelBaseNode *> &cache) {
  if (startNodeId == INVALID_TASK_ID) {
    return nullptr;
  }
  const auto *startNode = graph.GetNodeById(startNodeId);
  if (startNode == nullptr) {
    return nullptr;
  }
  const SkNodeType startNodeType = startNode->GetNodeType();
  uint64_t currentNodeId = startNodeId;
  std::vector<uint64_t> path;
  path.reserve(kMaxKernelSearchHops);
  for (int hop = 0; hop < kMaxKernelSearchHops; ++hop) {
    path.push_back(currentNodeId);
    const auto *current = graph.GetNodeById(currentNodeId);
    if (current == nullptr) {
      return nullptr;
    }
    const SuperKernelBaseNode *result = nullptr;
    const auto cached = cache.find(currentNodeId);
    if (cached != cache.end()) {
      result = cached->second;
    } else if (current->GetNodeType() == SkNodeType::NODE_KERNEL) {
      result = current;
    }
    if (result != nullptr) {
      for (uint64_t pathNodeId : path) {
        const auto *pathNode = graph.GetNodeById(pathNodeId);
        if (pathNode == nullptr) {
          return nullptr;
        }
        if (pathNode->GetNodeType() == SkNodeType::NODE_KERNEL || pathNode->GetNodeType() == startNodeType) {
          cache[pathNodeId] = result;
        }
      }
      return result;
    }
    currentNodeId = direction == SearchDirection::PREV ? current->GetPreNodeId() : current->GetNextNodeId();
    if (currentNodeId == INVALID_TASK_ID || std::find(path.cbegin(), path.cend(), currentNodeId) != path.cend()) {
      return nullptr;
    }
  }
  return nullptr;
}

SkQueueType ResolveMixWaitQueue(const SuperKernelBaseNode *previousKernel, const SuperKernelBaseNode &mixKernel) {
  if (previousKernel == nullptr) {
    return SkQueueType::AIV;
  }
  const SkQueueType previousQueue = ToQueueType(previousKernel->GetKernelType());
  if (previousQueue != SkQueueType::AIC && previousQueue != SkQueueType::AIV) {
    return SkQueueType::AIV;
  }
  const bool sameStream = previousKernel->GetStreamIdxInGraph() == mixKernel.GetStreamIdxInGraph();
  const bool directDependency = previousKernel->sendToNodeId.count(mixKernel.GetNodeId()) > 0 ||
                                mixKernel.receiveNodeId.count(previousKernel->GetNodeId()) > 0;
  if (sameStream || directDependency) {
    return previousQueue;
  }
  return previousQueue == SkQueueType::AIV ? SkQueueType::AIC : SkQueueType::AIV;
}

bool IsSupportedTask(const SuperKernelBaseNode &node) {
  const auto type = node.GetNodeType();
  return type == SkNodeType::NODE_KERNEL || type == SkNodeType::NODE_NOTIFY || type == SkNodeType::NODE_WAIT ||
         type == SkNodeType::NODE_RESET;
}
}  // namespace

bool SkQueueTypeResolver::Resolve(const std::vector<SuperKernelBaseNode *> &tasks,
                                  std::vector<SkQueueType> &queueTypes) const {
  queueTypes.clear();
  std::vector<SkQueueType> resolved(tasks.size(), SkQueueType::UNKNOWN);
  bool aicAvailable = false;
  bool aivAvailable = false;
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto *node = tasks[i];
    if (node == nullptr || !IsSupportedTask(*node)) {
      SK_LOGE("Resolve queue type failed: null or unsupported task node");
      return false;
    }
    if (node->GetNodeType() != SkNodeType::NODE_KERNEL) {
      continue;
    }
    resolved[i] = ToQueueType(node->GetKernelType());
    aicAvailable |= UsesAic(resolved[i]);
    aivAvailable |= UsesAiv(resolved[i]);
  }
  if (!aicAvailable && !aivAvailable) {
    SK_LOGE("Resolve queue type failed: no AIC or AIV resource detected in tasks");
    return false;
  }

  SkQueueType firstKernelEventQueue = SkQueueType::UNKNOWN;
  for (SkQueueType queueType : resolved) {
    if (queueType != SkQueueType::UNKNOWN) {
      firstKernelEventQueue = ToEventQueueType(queueType, aicAvailable, aivAvailable);
      break;
    }
  }

  std::unordered_map<uint64_t, const SuperKernelBaseNode *> cache;
  const SuperKernelBaseNode *previousScheduledKernel = nullptr;
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto *node = tasks[i];
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      previousScheduledKernel = node;
      continue;
    }
    const bool searchPrevious = node->GetNodeType() != SkNodeType::NODE_WAIT;
    const uint64_t anchor = searchPrevious ? node->GetPreNodeId() : node->GetNextNodeId();
    const auto *kernel = FindKernelNodeInDirection(
        anchor, graph_, searchPrevious ? SearchDirection::PREV : SearchDirection::NEXT, cache);
    if (kernel == nullptr) {
      resolved[i] = firstKernelEventQueue;
      continue;
    }
    cache[node->GetNodeId()] = kernel;
    SkQueueType queue = ToQueueType(kernel->GetKernelType());
    if (node->GetNodeType() == SkNodeType::NODE_WAIT && queue != SkQueueType::AIC && queue != SkQueueType::AIV) {
      queue = ResolveMixWaitQueue(previousScheduledKernel, *kernel);
    }
    resolved[i] = ToEventQueueType(queue, aicAvailable, aivAvailable);
  }
  queueTypes = std::move(resolved);
  return true;
}
