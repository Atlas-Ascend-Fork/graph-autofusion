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

#include <unordered_map>
#include <unordered_set>

#include "sk_graph.h"
#include "sk_log.h"
#include "sk_node.h"

namespace {
SkQueueType QueueTypeFromKernel(SkKernelType type) {
  switch (type) {
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
      SK_LOGW("unsupported kernel type %s for queue preparation, using AIC fallback", to_string(type));
      return SkQueueType::AIC;
  }
}

bool UsesAicQueue(SkQueueType type) {
  return type == SkQueueType::AIC || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

bool UsesAivQueue(SkQueueType type) {
  return type == SkQueueType::AIV || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

SkQueueType EventQueueType(SkQueueType type, bool aicAvailable, bool aivAvailable) {
  SkQueueType eventType = type == SkQueueType::AIC ? SkQueueType::AIC : SkQueueType::AIV;
  if (eventType == SkQueueType::AIC && !aicAvailable) {
    return SkQueueType::AIV;
  }
  if (eventType == SkQueueType::AIV && !aivAvailable) {
    return SkQueueType::AIC;
  }
  return eventType;
}

SkQueueType ResolveMixWaitEventQueueType(const SuperKernelBaseNode *previousKernel,
                                         const SuperKernelBaseNode *mixKernel) {
  if (previousKernel == nullptr) {
    return SkQueueType::AIV;
  }

  const SkQueueType previousKernelQueueType = QueueTypeFromKernel(previousKernel->GetKernelType());
  if (previousKernelQueueType != SkQueueType::AIC && previousKernelQueueType != SkQueueType::AIV) {
    return SkQueueType::AIV;
  }

  const bool sameStream = previousKernel->GetStreamIdxInGraph() == mixKernel->GetStreamIdxInGraph();
  const bool hasDirectDependency = previousKernel->sendToNodeId.count(mixKernel->GetNodeId()) > 0 ||
                                   mixKernel->receiveNodeId.count(previousKernel->GetNodeId()) > 0;
  return (sameStream || hasDirectDependency)
             ? previousKernelQueueType
             : ((previousKernelQueueType == SkQueueType::AIV) ? SkQueueType::AIC : SkQueueType::AIV);
}

const SuperKernelBaseNode *FindKernel(const SuperKernelGraph &graph, uint64_t startId, bool previous,
                                      std::unordered_map<uint64_t, const SuperKernelBaseNode *> &cache,
                                      int maxHops = 100) {
  if (startId == INVALID_TASK_ID) {
    return nullptr;
  }
  const auto *startNode = graph.GetNodeById(startId);
  if (startNode == nullptr) {
    return nullptr;
  }
  const SkNodeType startNodeType = startNode->GetNodeType();
  std::unordered_set<uint64_t> visited;
  std::vector<uint64_t> path;
  uint64_t currentId = startId;
  for (int hop = 0; hop < maxHops && currentId != INVALID_TASK_ID; ++hop) {
    if (!visited.insert(currentId).second) {
      return nullptr;
    }
    path.push_back(currentId);
    const auto *node = graph.GetNodeById(currentId);
    if (node == nullptr) {
      return nullptr;
    }
    auto cacheIt = cache.find(currentId);
    const SuperKernelBaseNode *kernel = cacheIt == cache.end() ? nullptr : cacheIt->second;
    if (kernel == nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      kernel = node;
    }
    if (kernel != nullptr) {
      for (uint64_t pathNodeId : path) {
        const auto *pathNode = graph.GetNodeById(pathNodeId);
        if (pathNode == nullptr) {
          return nullptr;
        }
        if (pathNode->GetNodeType() == SkNodeType::NODE_KERNEL || pathNode->GetNodeType() == startNodeType) {
          cache[pathNodeId] = kernel;
        }
      }
      return kernel;
    }
    currentId = previous ? node->GetPreNodeId() : node->GetNextNodeId();
  }
  return nullptr;
}
}  // namespace

bool SkQueueTypeResolver::Resolve(const std::vector<SuperKernelBaseNode *> &tasks) const {
  bool aicAvailable = false;
  bool aivAvailable = false;
  for (auto *node : tasks) {
    if (node == nullptr) {
      return false;
    }
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      SkQueueType type = QueueTypeFromKernel(node->GetKernelType());
      if (type == SkQueueType::UNKNOWN) {
        SK_LOGE("unsupported kernel queue type, nodeId=%lu", node->GetNodeId());
        return false;
      }
      node->SetExecutionQueueType(type);
      aicAvailable = aicAvailable || UsesAicQueue(type);
      aivAvailable = aivAvailable || UsesAivQueue(type);
    }
  }
  SkQueueType fallback = SkQueueType::UNKNOWN;
  for (auto *node : tasks) {
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      fallback = EventQueueType(node->GetExecutionQueueType(), aicAvailable, aivAvailable);
      break;
    }
  }
  if (fallback == SkQueueType::UNKNOWN) {
    return false;
  }

  SuperKernelBaseNode *previousKernel = nullptr;
  std::unordered_map<uint64_t, const SuperKernelBaseNode *> kernelNodeCache;
  for (auto *node : tasks) {
    const auto type = node->GetNodeType();
    if (type == SkNodeType::NODE_KERNEL) {
      previousKernel = node;
      continue;
    }
    if (type != SkNodeType::NODE_NOTIFY && type != SkNodeType::NODE_RESET && type != SkNodeType::NODE_WAIT) {
      SK_LOGE("unsupported node type for queue preparation, nodeId=%lu", node->GetNodeId());
      return false;
    }
    const bool previous = type != SkNodeType::NODE_WAIT;
    const auto *kernel =
        FindKernel(graph_, previous ? node->GetPreNodeId() : node->GetNextNodeId(), previous, kernelNodeCache);
    SkQueueType resolved = fallback;
    if (kernel != nullptr) {
      kernelNodeCache[node->GetNodeId()] = kernel;
      resolved = EventQueueType(QueueTypeFromKernel(kernel->GetKernelType()), aicAvailable, aivAvailable);
      if (type == SkNodeType::NODE_WAIT && (QueueTypeFromKernel(kernel->GetKernelType()) == SkQueueType::MIX_1_1 ||
                                            QueueTypeFromKernel(kernel->GetKernelType()) == SkQueueType::MIX_1_2)) {
        resolved = EventQueueType(ResolveMixWaitEventQueueType(previousKernel, kernel), aicAvailable, aivAvailable);
      }
    }
    node->SetExecutionQueueType(resolved);
  }
  return true;
}
