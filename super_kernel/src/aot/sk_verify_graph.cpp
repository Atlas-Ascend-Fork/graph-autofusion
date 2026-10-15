/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <memory>

#include "sk_verify_graph.h"
#include "sk_graph.h"
#include "sk_log.h"
#include "sk_node.h"
#include "sk_scope_split.h"

namespace {

enum class ScopeVerifyNodeFlagBitOffset : uint8_t {
  DYNAMIC_CORE = 0,
};

bool IsValidScopeVerifyNodeType(aclskScopeVerifyNodeType taskType) {
  return taskType == ACLSK_SCOPE_VERIFY_NODE_WAIT || taskType == ACLSK_SCOPE_VERIFY_NODE_NOTIFY ||
         taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE;
}

bool IsValidScopeVerifyKernelType(aclskScopeVerifyKernelType kernelType) {
  return kernelType == ACLSK_SCOPE_VERIFY_KERNEL_CUBE || kernelType == ACLSK_SCOPE_VERIFY_KERNEL_VECTOR ||
         kernelType == ACLSK_SCOPE_VERIFY_KERNEL_MIX;
}

bool IsScopeVerifyAiCoreOp(const aclskScopeVerifyNodeInfo &node) {
  // In scopeVerify, only compute nodes with positive numBlocks consume AiCore resources.
  return node.taskType == ACLSK_SCOPE_VERIFY_NODE_COMPUTE && node.numBlocks > 0;
}

bool IsDynamicScopeVerifyNode(const aclskScopeVerifyNodeInfo &node) {
  constexpr uint32_t dynamicCoreMask = 1U << static_cast<uint8_t>(ScopeVerifyNodeFlagBitOffset::DYNAMIC_CORE);
  return (node.flag & dynamicCoreMask) != 0;
}

bool ValidateScopeVerifyNode(const aclskScopeVerifyNodeInfo &node) {
  if (node.taskId < 0) {
    SK_DLOGE("aclskScopeVerify invalid taskId: %ld", node.taskId);
    return false;
  }
  if (node.streamId < 0 || node.streamId > std::numeric_limits<int32_t>::max()) {
    SK_DLOGE("aclskScopeVerify invalid streamId: %ld, taskId=%ld", node.streamId, node.taskId);
    return false;
  }
  if (!IsValidScopeVerifyNodeType(node.taskType)) {
    SK_DLOGE("aclskScopeVerify invalid taskType: %d, taskId=%ld", static_cast<int32_t>(node.taskType), node.taskId);
    return false;
  }
  if ((node.taskType == ACLSK_SCOPE_VERIFY_NODE_WAIT || node.taskType == ACLSK_SCOPE_VERIFY_NODE_NOTIFY) &&
      node.eventId < 0) {
    SK_DLOGE("aclskScopeVerify invalid eventId: %ld, taskId=%ld", node.eventId, node.taskId);
    return false;
  }
  if (IsScopeVerifyAiCoreOp(node) && !IsValidScopeVerifyKernelType(node.kernelType)) {
    SK_DLOGE("aclskScopeVerify invalid kernelType: %d, taskId=%ld", static_cast<int32_t>(node.kernelType), node.taskId);
    return false;
  }
  if (node.extendType != 0 || node.extendInfo != nullptr) {
    SK_DLOGE("aclskScopeVerify invalid node extend info, extendType=%d, extendInfo=%p, taskId=%ld", node.extendType,
             node.extendInfo, node.taskId);
    return false;
  }
  if (IsDynamicScopeVerifyNode(node)) {
    if (!IsScopeVerifyAiCoreOp(node)) {
      SK_DLOGE("aclskScopeVerify dynamic core limit requires AiCore compute node, taskId=%ld", node.taskId);
      return false;
    }
    if (node.coreLimit[0] < 0 || node.coreLimit[1] < 0) {
      SK_DLOGE("aclskScopeVerify invalid dynamic coreLimit={%d,%d}, taskId=%ld", node.coreLimit[0], node.coreLimit[1],
               node.taskId);
      return false;
    }
  }
  return true;
}

aclError ValidateScopeVerifyGraph(const aclskScopeVerifyGraphInfo *verifyGraph) {
  if (verifyGraph == nullptr) {
    SK_DLOGE("aclskScopeVerify graph is null");
    return ACL_ERROR_INVALID_PARAM;
  }
  if (verifyGraph->extendType != 0 || verifyGraph->extendInfo != nullptr) {
    SK_DLOGE("aclskScopeVerify invalid graph extend info, extendType=%d, extendInfo=%p", verifyGraph->extendType,
             verifyGraph->extendInfo);
    return ACL_ERROR_INVALID_PARAM;
  }
  if (verifyGraph->nodeCount > 0 && verifyGraph->nodes == nullptr) {
    SK_DLOGE("aclskScopeVerify graph nodes pointer is null, nodeCount=%zu", verifyGraph->nodeCount);
    return ACL_ERROR_INVALID_PARAM;
  }
  return ACL_SUCCESS;
}

bool FillScopeVerifyKernelInfos(const aclskScopeVerifyNodeInfo &verifyNode, KernelInfos &kernelInfos) {
  kernelInfos.kernelTypeInt = static_cast<uint32_t>(verifyNode.kernelType);
  kernelInfos.numBlocks = verifyNode.numBlocks;
  kernelInfos.taskRatio[0] = verifyNode.taskRatio[0];
  kernelInfos.taskRatio[1] = verifyNode.taskRatio[1];
  kernelInfos.isScheModeOn = verifyNode.scheMode != 0;

  switch (verifyNode.kernelType) {
    case ACLSK_SCOPE_VERIFY_KERNEL_CUBE:
      kernelInfos.kernelType = SkKernelType::AIC_ONLY;
      kernelInfos.cubeNum = verifyNode.numBlocks;
      kernelInfos.vecNum = 0;
      break;
    case ACLSK_SCOPE_VERIFY_KERNEL_VECTOR:
      kernelInfos.kernelType = SkKernelType::AIV_ONLY;
      kernelInfos.cubeNum = 0;
      kernelInfos.vecNum = verifyNode.numBlocks;
      break;
    case ACLSK_SCOPE_VERIFY_KERNEL_MIX: {
      uint32_t cubeRatio = verifyNode.taskRatio[0];
      uint32_t vecRatio = verifyNode.taskRatio[1];
      if (cubeRatio == 0 && vecRatio == 1) {
        kernelInfos.kernelType = SkKernelType::MIX_AIV_1_0;
      } else if (cubeRatio == 1 && vecRatio == 0) {
        kernelInfos.kernelType = SkKernelType::MIX_AIC_1_0;
      } else if (cubeRatio == 1 && vecRatio == 1) {
        kernelInfos.kernelType = SkKernelType::MIX_AIC_1_1;
      } else if (cubeRatio == 1 && vecRatio == 2) {
        kernelInfos.kernelType = SkKernelType::MIX_AIC_1_2;
      } else {
        SK_DLOGE("aclskScopeVerify invalid MIX taskRatio, taskId=%ld, ratio={%u,%u}", verifyNode.taskId, cubeRatio,
                 vecRatio);
        return false;
      }
      kernelInfos.cubeNum = verifyNode.numBlocks * cubeRatio;
      kernelInfos.vecNum = verifyNode.numBlocks * vecRatio;
      break;
    }
    default:
      return false;
  }

  if (IsDynamicScopeVerifyNode(verifyNode)) {
    const uint32_t originalCubeNum = kernelInfos.cubeNum;
    const uint32_t originalVecNum = kernelInfos.vecNum;
    kernelInfos.cubeNum = static_cast<uint32_t>(verifyNode.coreLimit[0]);
    kernelInfos.vecNum = static_cast<uint32_t>(verifyNode.coreLimit[1]);
    SK_DLOGI(
        "[aclskScopeVerify] dynamic core override, taskId=%ld, flag=0x%x, numBlocks=%u, "
        "cubeNum=%u->%d, vecNum=%u->%d",
        verifyNode.taskId, verifyNode.flag, verifyNode.numBlocks, originalCubeNum, verifyNode.coreLimit[0],
        originalVecNum, verifyNode.coreLimit[1]);
  }
  return true;
}

std::unique_ptr<SuperKernelBaseNode> CreateScopeVerifyNode(const aclskScopeVerifyNodeInfo &verifyNode,
                                                           uint64_t nodeIdxInStream, uint32_t streamIdxInGraph,
                                                           uint64_t preNodeId) {
  uint64_t nodeId = static_cast<uint64_t>(verifyNode.taskId);
  int32_t streamId = static_cast<int32_t>(verifyNode.streamId);
  std::unique_ptr<SuperKernelBaseNode> node;

  if (IsScopeVerifyAiCoreOp(verifyNode)) {
    node = std::make_unique<SuperKernelKernelNode>(nullptr, ACL_MODEL_RI_TASK_KERNEL, nodeIdxInStream, streamIdxInGraph,
                                                   streamId, preNodeId);
    node->SetNodeType(SkNodeType::NODE_KERNEL);
    if (!FillScopeVerifyKernelInfos(verifyNode, node->nodeInfos.kernelInfos)) {
      return nullptr;
    }
  } else if (verifyNode.taskType == ACLSK_SCOPE_VERIFY_NODE_NOTIFY) {
    node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WRITE, nodeIdxInStream,
                                                   streamIdxInGraph, streamId, preNodeId);
    node->SetNodeType(SkNodeType::NODE_NOTIFY);
    node->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(verifyNode.eventId);
  } else if (verifyNode.taskType == ACLSK_SCOPE_VERIFY_NODE_WAIT) {
    node = std::make_unique<SuperKernelMemoryNode>(nullptr, ACL_MODEL_RI_TASK_VALUE_WAIT, nodeIdxInStream,
                                                   streamIdxInGraph, streamId, preNodeId);
    node->SetNodeType(SkNodeType::NODE_WAIT);
    node->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(verifyNode.eventId);
  } else {
    node = std::make_unique<SuperKernelDefaultNode>(nullptr, ACL_MODEL_RI_TASK_DEFAULT, nodeIdxInStream,
                                                    streamIdxInGraph, streamId, preNodeId);
    node->SetNodeType(SkNodeType::NODE_DEFAULT);
  }

  node->SetNodeId(nodeId);
  return node;
}

void RebuildScopeVerifyStreamInfos(std::vector<SuperKernelScopeInfo> &initialScopes) {
  for (auto &scope : initialScopes) {
    ScopeSplitPass::RebuildStreamInfos(scope);
  }
}

}  // namespace

void ScopeVerifyGraphBuilder::ResetBuildState() {
  streamIdxById_.clear();
  scopeIdxById_.clear();
  preNodeIds_.clear();
}

aclError ScopeVerifyGraphBuilder::ResolveStream(SuperKernelGraph &graph, int64_t streamId, uint32_t &streamIdx) {
  auto streamIt = streamIdxById_.find(streamId);
  if (streamIt != streamIdxById_.end()) {
    streamIdx = streamIt->second;
    return ACL_SUCCESS;
  }

  if (graph.streams.size() > std::numeric_limits<uint32_t>::max()) {
    SK_DLOGE("aclskScopeVerify stream count exceeds uint32 range, size=%zu", graph.streams.size());
    return ACL_ERROR_INVALID_PARAM;
  }

  streamIdx = static_cast<uint32_t>(graph.streams.size());
  streamIdxById_[streamId] = streamIdx;
  graph.streams.emplace_back(nullptr);
  graph.nodeSizeInStream.emplace_back(0);
  if (preNodeIds_.size() <= streamIdx) {
    preNodeIds_.resize(static_cast<size_t>(streamIdx) + 1U, INVALID_TASK_ID);
  }
  return ACL_SUCCESS;
}

SuperKernelScopeInfo &ScopeVerifyGraphBuilder::ResolveScope(int32_t scopeId,
                                                            std::vector<SuperKernelScopeInfo> &initialScopes) {
  auto scopeIt = scopeIdxById_.find(scopeId);
  if (scopeIt != scopeIdxById_.end()) {
    return initialScopes[scopeIt->second];
  }

  size_t scopeIdx = initialScopes.size();
  scopeIdxById_[scopeId] = scopeIdx;
  initialScopes.emplace_back();
  return initialScopes[scopeIdx];
}

aclError ScopeVerifyGraphBuilder::AppendNode(
    SuperKernelGraph &graph, aclskScopeVerifyNodeInfo &verifyNode, std::vector<SuperKernelScopeInfo> &initialScopes,
    std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> &verifyNodesById) {
  if (!ValidateScopeVerifyNode(verifyNode)) {
    return ACL_ERROR_INVALID_PARAM;
  }

  uint64_t nodeId = static_cast<uint64_t>(verifyNode.taskId);
  if (verifyNodesById.find(nodeId) != verifyNodesById.end()) {
    SK_DLOGE("aclskScopeVerify duplicate taskId: %lu", nodeId);
    return ACL_ERROR_INVALID_PARAM;
  }

  uint32_t streamIdx = 0;
  aclError ret = ResolveStream(graph, verifyNode.streamId, streamIdx);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  uint64_t nodeIdxInStream = graph.nodeSizeInStream[streamIdx];
  uint64_t preNodeId = preNodeIds_[streamIdx];
  auto node = CreateScopeVerifyNode(verifyNode, nodeIdxInStream, streamIdx, preNodeId);
  if (node == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }

  SuperKernelBaseNode *nodePtr = node.get();
  if (!graph.AddNode(std::move(node))) {
    return ACL_ERROR_INVALID_PARAM;
  }
  graph.UpdateNodeRelations(nodeId, streamIdx, nodeIdxInStream, preNodeIds_[streamIdx]);
  graph.nodeSizeInStream[streamIdx]++;

  verifyNodesById[nodeId] = &verifyNode;
  // Negative scopeId means the task is outside SK scope. Keep it in graph, but do not add it to initial scopes.
  if (verifyNode.scopeId >= 0) {
    ResolveScope(verifyNode.scopeId, initialScopes).AddNode(nodePtr);
  }
  return ACL_SUCCESS;
}

aclError ScopeVerifyGraphBuilder::Build(const aclskScopeVerifyGraphInfo *verifyGraph, SuperKernelGraph &graph,
                                        std::vector<SuperKernelScopeInfo> &initialScopes,
                                        std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> &verifyNodesById) {
  ResetBuildState();

  aclError ret = ValidateScopeVerifyGraph(verifyGraph);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  for (size_t i = 0; i < verifyGraph->nodeCount; ++i) {
    ret = AppendNode(graph, verifyGraph->nodes[i], initialScopes, verifyNodesById);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }

  graph.BuildEventNodeAssociations();
  RebuildScopeVerifyStreamInfos(initialScopes);
  return ACL_SUCCESS;
}
