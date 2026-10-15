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
 * \file sk_lock_detector.cpp
 * \brief
 */

#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_common.h"
#include "super_kernel.h"
#include <functional>
#include <map>

int64_t LockDetector::deviceRealCubeNum = 0;
int64_t LockDetector::deviceRealVecNum = 0;

void LockDetector::Init(SuperKernelGraph &graph) {
  SK_LOGD("[lock detector] LockDetector::Init: Initializing lock detector");
  nodes.clear();
  depOpCubeNum = 0;
  depOpVecNum = 0;
  superKernelCubeNum = 0;
  superKernelVecNum = 0;
  currentScopeId_ = INVALID_SCOPE_ID;
  fusedNotifyWaitInfos_.clear();
  scopeCoreInfoChanged_ = false;
  nodeNum = 0;
  kernelNodeNum = 0;
  skStreamIds.clear();
  parallelScopeHeadNodeIds_.clear();
  parallelScopeTailNodeIds_.clear();
  graph_ = &graph;

  if (deviceRealCubeNum == 0 && deviceRealVecNum == 0) {
    GetDeviceCores();
  } else {
    SK_LOGD("[lock detector] LockDetector::Init: Using cached device cores: cube=%lu, vec=%lu", deviceRealCubeNum,
            deviceRealVecNum);
  }
}

aclError LockDetector::GetDeviceCores() {
  if (deviceRealCubeNum != 0 && deviceRealVecNum != 0) {
    return ACL_SUCCESS;
  }

  aclError ret = GetDeviceCoreNums(deviceRealCubeNum, deviceRealVecNum);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("[lock detector] GetDeviceCores failed, ret=%d", ret);
    return ret;
  }

  SK_LOGI("[lock detector] GetDeviceCores success, cube=%u, vec=%u", deviceRealCubeNum, deviceRealVecNum);
  return ACL_SUCCESS;
}

std::pair<uint64_t, uint64_t> LockDetector::GetAvailableCores(bool isSuperKernel) const {
  const auto getRemainingCores = [](int64_t deviceCores, uint32_t occupiedCores) -> uint64_t {
    if (deviceCores <= 0 || occupiedCores >= static_cast<uint64_t>(deviceCores)) {
      return 0;
    }
    return deviceCores - occupiedCores;
  };
  if (isSuperKernel) {
    return {getRemainingCores(deviceRealCubeNum, depOpCubeNum), getRemainingCores(deviceRealVecNum, depOpVecNum)};
  } else {
    return {getRemainingCores(deviceRealCubeNum, superKernelCubeNum),
            getRemainingCores(deviceRealVecNum, superKernelVecNum)};
  }
}

bool LockDetector::HasEnoughDeviceCores(uint64_t skCubeNum, uint64_t skVecNum, uint64_t depCubeNum,
                                        uint64_t depVecNum) const {
  if (deviceRealCubeNum < 0 || deviceRealVecNum < 0 || depCubeNum > static_cast<uint64_t>(deviceRealCubeNum) ||
      depVecNum > static_cast<uint64_t>(deviceRealVecNum)) {
    return false;
  }
  return skCubeNum <= deviceRealCubeNum - depCubeNum && skVecNum <= deviceRealVecNum - depVecNum;
}

void LockDetector::SetScopeInfos(const std::vector<SuperKernelScopeInfo> &scopes) {
  scopeInfos_.clear();
  for (const auto &scope : scopes) {
    UpdateScopeInfo(scope);
  }
}

void LockDetector::UpdateScopeInfo(const SuperKernelScopeInfo &scope) {
  if (!scope.GetScopeCoreInfo().IsValid()) {
    scopeInfos_.erase(scope.GetScopeId());
    return;
  }

  ScopeRuntimeInfo scopeInfo;
  scopeInfo.coreInfo = scope.GetScopeCoreInfo();
  for (const auto &streamInfo : scope.GetScopeStreamInfos()) {
    scopeInfo.streamIds.insert(streamInfo.streamIdx);
    scopeInfo.headNodeIds.emplace(streamInfo.streamIdx, streamInfo.headNodeIdx);
    scopeInfo.tailNodeIds.emplace(streamInfo.streamIdx, streamInfo.tailNodeIdx);
  }
  scopeInfos_[scope.GetScopeId()] = std::move(scopeInfo);
}

void LockDetector::RemoveScopeInfo(uint16_t scopeId) {
  scopeInfos_.erase(scopeId);
}

bool LockDetector::IsNodeBeforeCandidateScope(const SuperKernelBaseNode &node,
                                              const ScopeRuntimeInfo &candidateScope) const {
  const auto headIt = candidateScope.headNodeIds.find(node.GetStreamIdxInGraph());
  if (headIt == candidateScope.headNodeIds.end()) {
    return false;
  }
  const auto *headNode = graph_->GetNodeById(headIt->second);
  return headNode != nullptr && node.GetNodeIdxInStream() < headNode->GetNodeIdxInStream();
}

bool LockDetector::IsNodeAfterCandidateScope(const SuperKernelBaseNode &node,
                                             const ScopeRuntimeInfo &candidateScope) const {
  const auto tailIt = candidateScope.tailNodeIds.find(node.GetStreamIdxInGraph());
  if (tailIt == candidateScope.tailNodeIds.end()) {
    return false;
  }
  const auto *tailNode = graph_->GetNodeById(tailIt->second);
  return tailNode != nullptr && node.GetNodeIdxInStream() > tailNode->GetNodeIdxInStream();
}

bool LockDetector::IsSerializedWithCandidateScope(const SuperKernelBaseNode &waitNode, uint64_t candidateWaitNodeId,
                                                  const ScopeRuntimeInfo &candidateScope) const {
  for (const auto &headNodeEntry : parallelScopeHeadNodeIds_) {
    const uint64_t headNodeId = headNodeEntry.second;
    auto *scopeHead = graph_->GetNodeById(headNodeId);
    if (scopeHead == nullptr) {
      continue;
    }
    uint64_t nodeId = scopeHead->GetPreNodeId();
    while (nodeId != INVALID_TASK_ID) {
      auto *node = graph_->GetNodeById(nodeId);
      if (node == nullptr) {
        break;
      }
      if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
        auto *notifyNode = graph_->GetNodeById(node->GetCorrespondingNotifyNodeId());
        if (notifyNode != nullptr && IsNodeAfterCandidateScope(*notifyNode, candidateScope)) {
          SK_LOGI("[lock detector] Candidate Wait %lu scope is serialized before current SK by notify %lu -> wait %lu",
                  candidateWaitNodeId, notifyNode->GetNodeId(), node->GetNodeId());
          return true;
        }
      }
      nodeId = node->GetPreNodeId();
    }
  }

  for (const auto &[streamId, tailNodeId] : parallelScopeTailNodeIds_) {
    uint64_t nodeId = INVALID_TASK_ID;
    if (streamId == waitNode.GetStreamIdxInGraph()) {
      nodeId = waitNode.GetNextNodeId();
    } else {
      auto *scopeTail = graph_->GetNodeById(tailNodeId);
      if (scopeTail != nullptr) {
        nodeId = scopeTail->GetNextNodeId();
      }
    }
    while (nodeId != INVALID_TASK_ID) {
      auto *node = graph_->GetNodeById(nodeId);
      if (node == nullptr) {
        break;
      }
      if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
        for (const auto candidateWaitId : node->GetCorrespondingWaitNodeIds()) {
          auto *candidateWait = graph_->GetNodeById(candidateWaitId);
          if (candidateWait != nullptr && IsNodeBeforeCandidateScope(*candidateWait, candidateScope)) {
            SK_LOGI(
                "[lock detector] Current SK is serialized before candidate Wait %lu scope by notify %lu -> wait %lu",
                candidateWaitNodeId, node->GetNodeId(), candidateWait->GetNodeId());
            return true;
          }
        }
      }
      nodeId = node->GetNextNodeId();
    }
  }
  return false;
}

void LockDetector::MergeCandidateScopeRange(const ScopeRuntimeInfo &candidateScope) {
  for (const auto streamId : candidateScope.streamIds) {
    const auto candidateHeadIt = candidateScope.headNodeIds.find(streamId);
    const auto candidateTailIt = candidateScope.tailNodeIds.find(streamId);
    if (candidateHeadIt == candidateScope.headNodeIds.end() || candidateTailIt == candidateScope.tailNodeIds.end()) {
      continue;
    }
    const auto *candidateHeadNode = graph_->GetNodeById(candidateHeadIt->second);
    const auto *candidateTailNode = graph_->GetNodeById(candidateTailIt->second);
    if (candidateHeadNode == nullptr || candidateTailNode == nullptr) {
      continue;
    }
    UpdateSKRangeInStream(streamId, *candidateHeadNode, *candidateTailNode);
  }
}

bool LockDetector::CheckNotifyWaitScopeCombinations(const SuperKernelBaseNode &waitNode,
                                                    const SuperKernelBaseNode &notifyNode, uint32_t skCubeNum,
                                                    uint32_t skVecNum, uint32_t depCubeNum, uint32_t depVecNum,
                                                    const std::unordered_set<uint32_t> &predecessorStreamIds) {
  std::map<uint16_t, uint64_t> candidateWaitsByScope;
  for (const auto candidateWaitNodeId : notifyNode.GetCorrespondingWaitNodeIds()) {
    const auto *candidateWaitNode = graph_->GetNodeById(candidateWaitNodeId);
    if (candidateWaitNode == nullptr || candidateWaitNode->GetScopeId() == INVALID_SCOPE_ID ||
        candidateWaitNode->GetScopeId() == currentScopeId_) {
      continue;
    }
    const auto scopeInfoIt = scopeInfos_.find(candidateWaitNode->GetScopeId());
    if (scopeInfoIt == scopeInfos_.end()) {
      continue;
    }
    candidateWaitsByScope.emplace(candidateWaitNode->GetScopeId(), candidateWaitNodeId);
  }
  std::vector<std::pair<uint64_t, const ScopeRuntimeInfo *>> candidates;
  for (const auto &[scopeId, candidateWaitId] : candidateWaitsByScope) {
    candidates.emplace_back(candidateWaitId, &scopeInfos_.at(scopeId));
  }

  std::function<bool(size_t, uint64_t, uint64_t)> check = [&](size_t index, uint64_t cubeNum, uint64_t vecNum) {
    if (!HasEnoughDeviceCores(cubeNum, vecNum, depCubeNum, depVecNum)) {
      return false;
    }
    if (index == candidates.size()) {
      return true;
    }
    if (!check(index + 1, cubeNum, vecNum)) {
      return false;
    }
    const auto &[candidateWaitId, candidateScope] = candidates[index];
    if (HasIntersection(candidateScope->streamIds, skStreamIds) ||
        HasIntersection(candidateScope->streamIds, predecessorStreamIds) ||
        IsSerializedWithCandidateScope(waitNode, candidateWaitId, *candidateScope)) {
      return true;
    }

    const auto savedStreamIds = skStreamIds;
    const auto savedRanges = skRangeInStream;
    const auto savedHeads = parallelScopeHeadNodeIds_;
    const auto savedTails = parallelScopeTailNodeIds_;
    skStreamIds.insert(candidateScope->streamIds.begin(), candidateScope->streamIds.end());
    MergeCandidateScopeRange(*candidateScope);
    const bool valid = check(index + 1, cubeNum + candidateScope->coreInfo.GetCubeNum(),
                             vecNum + candidateScope->coreInfo.GetVectorNum());
    skStreamIds = savedStreamIds;
    skRangeInStream = savedRanges;
    parallelScopeHeadNodeIds_ = savedHeads;
    parallelScopeTailNodeIds_ = savedTails;
    return valid;
  };
  return check(0, skCubeNum, skVecNum);
}

bool LockDetector::CheckFusedWaitNotifies(uint32_t skCubeNum, uint32_t skVecNum, uint32_t depCubeNum,
                                          uint32_t depVecNum) {
  for (const auto &[notifyId, waitInfo] : fusedNotifyWaitInfos_) {
    const auto *notifyNode = graph_->GetNodeById(notifyId);
    const auto *waitNode = graph_->GetNodeById(waitInfo.waitId);
    if (notifyNode != nullptr && waitNode != nullptr &&
        !CheckNotifyWaitScopeCombinations(*waitNode, *notifyNode, skCubeNum, skVecNum, depCubeNum, depVecNum,
                                          waitInfo.predecessorStreamIds)) {
      deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
      return false;
    }
  }
  return true;
}

bool LockDetector::IsInSKStream(const SuperKernelBaseNode &node) const {
  return std::find(skStreamIds.begin(), skStreamIds.end(), node.GetStreamIdxInGraph()) != skStreamIds.end();
}

bool LockDetector::HasDeadlock(SuperKernelBaseNode *curNode) {
  curNode->SetVisited(true);
  tempVisitedNodes.emplace_back(curNode->GetNodeId());
  if (curNode->GetPreNodeId() == INVALID_TASK_ID) {
    return false;
  }

  const uint64_t preNodeId = curNode->GetPreNodeId();
  SuperKernelBaseNode *preNode = graph_->GetNodeById(preNodeId);
  if (preNode == nullptr) {
    SK_LOGE("[lock detector] HasDeadlock: preNode %lu not found for curNode %lu", preNodeId, curNode->GetNodeId());
    return false;
  }
  // if prenode is visited, means that wait node has already entered the detection process before
  // when finish in this condition, there are two situations:
  //      Case 1: one branch is checked forward
  //      Case 2: two branches (from wait node in graph) are checked forward, which ensure both branches return false in
  //      wait
  if (preNode->IsVisited()) {
    return false;
  }

  // when prenode has been fused in sk
  // if streams of sk with prenode intersects with the stream of current sk, means prenode has been executed
  if (HasIntersection(preNode->GetScopeStreamIds(), skStreamIds)) {
    return false;
  }

  bool hasDeadlock = true;
  switch (preNode->GetNodeType()) {
    case SkNodeType::NODE_KERNEL:
      hasDeadlock = preNode->IsScopeNode() ? HasDeadlock(preNode) : CheckKernelNodeDeadlock(preNode);
      break;
    case SkNodeType::NODE_WAIT:
      hasDeadlock = CheckWaitNodeDeadlock(preNode);
      break;
    case SkNodeType::NODE_NOTIFY:
      hasDeadlock = CheckNotifyNodeDeadlock(preNode);
      break;
    case SkNodeType::NODE_DEFAULT:
      hasDeadlock = HasDeadlock(preNode);
      break;
    case SkNodeType::NODE_RESET:
      hasDeadlock = HasDeadlock(preNode);
      break;
    default:
      SK_LOGD("nodeId: %u, unsupported node type %u in HasDeadlock", preNode->GetNodeId(), preNode->GetNodeType());
      deadlockReason_ = DeadlockFailReason::NO_SUPPORT_NODE;
      break;
  }

  return hasDeadlock;
}

bool LockDetector::CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode) {
  if (!HasEnoughCores(preNode, false)) {
    SK_LOGI("Not enough cores for kernel node, nodeId=%lu, requiredCube=%u, requiredVec=%u", preNode->GetNodeId(),
            preNode->GetCubeNum(), preNode->GetVecNum());
    deadlockReason_ = DeadlockFailReason::KERNEL_INSUFFICIENT_CORES;
    return true;
  }
  if (HasDeadlock(preNode)) {
    SK_LOGI("Deadlock detected in kernel node, nodeId=%lu", preNode->GetNodeId());
    return true;
  }
  return false;
}

bool LockDetector::CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode) {
  const uint64_t notifyId = preNode->GetCorrespondingNotifyNodeId();
  // Case 1: notify node not in modelRI
  if (notifyId == INVALID_TASK_ID) {
    SK_LOGI("Deadlock detected in wait node, waitNodeId=%lu, notifyNodeId=%lu is not in graph", preNode->GetNodeId(),
            notifyId);
    deadlockReason_ = DeadlockFailReason::NOTIFY_NOT_IN_GRAPH;
    return true;
  }
  SuperKernelBaseNode *notifyNode = graph_->GetNodeById(notifyId);
  // abnormal case, notify node not found
  if (notifyNode == nullptr) {
    SK_LOGE("[lock detector] CheckWaitNodeDeadlock: notifyNode %lu not found for waitNode %lu", notifyId,
            preNode->GetNodeId());
    deadlockReason_ = DeadlockFailReason::NOTIFY_INVALID;
    return true;
  }
  // Case 2: notify node is after sk range,
  if (IsAfterSKRange(*notifyNode)) {
    deadlockReason_ = DeadlockFailReason::NOTIFY_AFTER_SK_RANGE;
    return true;
  }
  // Case 3: check node before wait node in current stream
  if (HasDeadlock(preNode)) {
    SK_LOGI("Deadlock detected in wait node pre-path, waitNodeId=%lu", preNode->GetNodeId());
    return true;
  }
  // Case 4: check node before notify node (notify node not in sk stream) in different stream
  if (notifyNode->GetStreamIdxInGraph() != preNode->GetStreamIdxInGraph() && !IsInSKStream(*notifyNode)) {
    if (HasDeadlock(notifyNode)) {
      SK_LOGI("Deadlock detected in wait node cross-stream path, waitNodeId=%lu, notifyNodeId=%lu",
              preNode->GetNodeId(), notifyNode->GetNodeId());
      return true;
    }
  }
  return false;
}

bool LockDetector::CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode) {
  const uint32_t cubeNum = preNode->GetCubeNum();
  const uint32_t vecNum = preNode->GetVecNum();
  if ((cubeNum > 0 || vecNum > 0) && !HasEnoughCores(preNode, false)) {
    SK_LOGI("Not enough cores for notify node, nodeId=%lu, requiredCube=%u, requiredVec=%u", preNode->GetNodeId(),
            cubeNum, vecNum);
    deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
    return true;
  }
  std::vector<uint64_t> waitIds = preNode->GetCorrespondingWaitNodeIds();
  for (uint64_t waitId : waitIds) {
    SuperKernelBaseNode *waitNode = graph_->GetNodeById(waitId);
    // if exist wait node before sk, means notify has been executed, no need to check
    if (IsBeforeSKRange(*waitNode)) {
      break;
    } else {
      if (HasDeadlock(preNode)) {
        SK_LOGI("Deadlock detected in notify node path, notifyNodeId=%lu", preNode->GetNodeId());
        return true;
      }
    }
  }
  return false;
}

bool LockDetector::HasEnoughCores(const SuperKernelBaseNode *curNode, bool isSuperKernel) {
  uint32_t curNodeCubeNum = curNode->GetCubeNum();
  uint32_t curNodeVecNum = curNode->GetVecNum();
  if (currentScopeId_ != INVALID_SCOPE_ID && curNode->GetScopeId() == currentScopeId_) {
    curNodeCubeNum = superKernelCubeNum;
    curNodeVecNum = superKernelVecNum;
  } else {
    const auto scopeInfoIt = scopeInfos_.find(curNode->GetScopeId());
    if (scopeInfoIt != scopeInfos_.end() && scopeInfoIt->second.coreInfo.IsValid()) {
      curNodeCubeNum = scopeInfoIt->second.coreInfo.GetCubeNum();
      curNodeVecNum = scopeInfoIt->second.coreInfo.GetVectorNum();
    }
  }

  if (isSuperKernel) {
    if (!scopeCoreInfoChanged_) {
      SK_LOGD("[lock detector] Candidate SK core requirement is unchanged for node %s, skip resource recheck",
              curNode->Format().c_str());
      return true;
    }
    // depOp cores retain the resource requirements discovered from preceding Wait/Notify paths.
    const auto availableCores = GetAvailableCores(true);
    if (HasEnoughDeviceCores(superKernelCubeNum, superKernelVecNum, depOpCubeNum, depOpVecNum)) {
      SK_LOGD("[lock detector] Candidate SK for node %s fits available cores (cube %u<=%u, vec %u<=%u)",
              curNode->Format().c_str(), superKernelCubeNum, availableCores.first, superKernelVecNum,
              availableCores.second);
      return CheckFusedWaitNotifies(superKernelCubeNum, superKernelVecNum, depOpCubeNum, depOpVecNum);
    }
    SK_LOGD("[lock detector] Candidate SK for node %s conflicts with dependent ops (cube %u>%u or vec %u>%u)",
            curNode->Format().c_str(), superKernelCubeNum, availableCores.first, superKernelVecNum,
            availableCores.second);
    return false;
  } else {
    const uint32_t candidateDepCubeNum = std::max(depOpCubeNum, curNodeCubeNum);
    const uint32_t candidateDepVecNum = std::max(depOpVecNum, curNodeVecNum);
    std::pair<uint32_t, uint32_t> availableCores = GetAvailableCores(false);
    if (HasEnoughDeviceCores(superKernelCubeNum, superKernelVecNum, candidateDepCubeNum, candidateDepVecNum)) {
      if (!CheckFusedWaitNotifies(superKernelCubeNum, superKernelVecNum, candidateDepCubeNum, candidateDepVecNum)) {
        return false;
      }
      depOpCubeNum = candidateDepCubeNum;
      depOpVecNum = candidateDepVecNum;
      SK_LOGD("[lock detector] Node %s: allocated from device (cube %u, vec %u), new depOp limits: cube %u, vec %u",
              curNode->Format().c_str(), curNodeCubeNum, curNodeVecNum, depOpCubeNum, depOpVecNum);
      return true;
    }
    SK_LOGD(
        "[lock detector] Node %s: insufficient cores for depOp (required: cube %u, vec %u, available: cube %u, "
        "vec %u)",
        curNode->Format().c_str(), curNodeCubeNum, curNodeVecNum, availableCores.first, availableCores.second);
    return false;
  }
}

void LockDetector::RollbackVisitedState(std::vector<uint64_t> &visitedNodes) {
  for (auto nodeId : visitedNodes) {
    SuperKernelBaseNode *node = graph_->GetNodeById(nodeId);
    if (node != nullptr) {
      node->SetVisited(false);
    }
  }
  visitedNodes.clear();
}

void LockDetector::Reset() {
  SK_LOGD("[lock detector] LockDetector::Reset: Resetting lock detector state");
  SK_LOGD(
      "[lock detector] Previous state: depOpCubeNum=%u, depOpVecNum=%u, superKernelCubeNum=%u, superKernelVecNum=%u, "
      "nodeNum=%u, kernelNodeNum=%u",
      depOpCubeNum, depOpVecNum, superKernelCubeNum, superKernelVecNum, nodeNum, kernelNodeNum);

  const size_t nodesCount = nodes.size();
  const size_t tempVisitedNodesCount = tempVisitedNodes.size();
  const size_t streamIdsCount = skStreamIds.size();
  const size_t streamRangesCount = skRangeInStream.size();

  depOpCubeNum = 0;
  depOpVecNum = 0;
  superKernelCubeNum = 0;
  superKernelVecNum = 0;
  currentScopeId_ = INVALID_SCOPE_ID;
  fusedNotifyWaitInfos_.clear();
  scopeCoreInfoChanged_ = false;
  nodeNum = 0;
  kernelNodeNum = 0;
  deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;

  RollbackVisitedState(nodes);
  RollbackVisitedState(tempVisitedNodes);
  skStreamIds.clear();
  skRangeInStream.clear();
  parallelScopeHeadNodeIds_.clear();
  parallelScopeTailNodeIds_.clear();
  SK_LOGD(
      "[lock detector] Reset: Completed, cleared %zu nodes, %zu tempVisitedNodes, %zu stream IDs, %zu stream ranges",
      nodesCount, tempVisitedNodesCount, streamIdsCount, streamRangesCount);
}

void LockDetector::UpdateSKRangeInStream(const SuperKernelBaseNode &curNode) {
  UpdateSKRangeInStream(curNode.GetStreamIdxInGraph(), curNode, curNode);
}

void LockDetector::UpdateSKRangeInStream(uint32_t streamId, const SuperKernelBaseNode &headNode,
                                         const SuperKernelBaseNode &tailNode) {
  const uint64_t headPosition = headNode.GetNodeIdxInStream();
  const uint64_t tailPosition = tailNode.GetNodeIdxInStream();
  auto rangeIt = skRangeInStream.find(streamId);
  if (rangeIt == skRangeInStream.end()) {
    skRangeInStream.emplace(streamId, std::make_pair(headPosition, tailPosition));
  } else {
    rangeIt->second.first = std::min(rangeIt->second.first, headPosition);
    rangeIt->second.second = std::max(rangeIt->second.second, tailPosition);
  }

  auto headIt = parallelScopeHeadNodeIds_.find(streamId);
  if (headIt == parallelScopeHeadNodeIds_.end()) {
    parallelScopeHeadNodeIds_.emplace(streamId, headNode.GetNodeId());
  } else {
    const auto *currentHeadNode = graph_->GetNodeById(headIt->second);
    if (currentHeadNode == nullptr || headPosition < currentHeadNode->GetNodeIdxInStream()) {
      headIt->second = headNode.GetNodeId();
    }
  }
  auto tailIt = parallelScopeTailNodeIds_.find(streamId);
  if (tailIt == parallelScopeTailNodeIds_.end()) {
    parallelScopeTailNodeIds_.emplace(streamId, tailNode.GetNodeId());
  } else {
    const auto *currentTailNode = graph_->GetNodeById(tailIt->second);
    if (currentTailNode == nullptr || tailPosition > currentTailNode->GetNodeIdxInStream()) {
      tailIt->second = tailNode.GetNodeId();
    }
  }
}

bool LockDetector::IsBeforeSKRange(const SuperKernelBaseNode &curNode) {
  const uint64_t nodeId = curNode.GetNodeIdxInStream();
  uint32_t streamId = curNode.GetStreamIdxInGraph();
  if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
    return false;
  }
  return nodeId < skRangeInStream[streamId].first;
}

bool LockDetector::IsAfterSKRange(const SuperKernelBaseNode &curNode) {
  uint64_t nodeId = curNode.GetNodeIdxInStream();
  uint32_t streamId = curNode.GetStreamIdxInGraph();
  if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
    return false;
  }
  return nodeId > skRangeInStream[streamId].second;
}

bool LockDetector::HasIntersection(const std::unordered_set<uint32_t> &lhsStreams,
                                   const std::unordered_set<uint32_t> &rhsStreams) const {
  for (const auto &streamId : lhsStreams) {
    if (rhsStreams.count(streamId) > 0) {
      return true;
    }
  }
  return false;
}

bool LockDetector::GetWaitNodeFusibleStatus(SuperKernelBaseNode &curNode) {
  uint64_t notifyId = curNode.GetCorrespondingNotifyNodeId();
  // Case 1: notify node not in modelRI
  if (notifyId == INVALID_TASK_ID) {
    SK_LOGD("[lock detector] Wait node %s: notify node %lu not found in graph", curNode.Format().c_str(), notifyId);
    deadlockReason_ = DeadlockFailReason::NOTIFY_NOT_IN_GRAPH;
    return false;
  }
  SuperKernelBaseNode *notifyNode = graph_->GetNodeById(notifyId);
  // abnormal case: notify node not found
  if (notifyNode == nullptr) {
    SK_LOGE("[lock detector] Wait node %s: notify node %lu not found in graph", curNode.Format().c_str(), notifyId);
    deadlockReason_ = DeadlockFailReason::NOTIFY_INVALID;
    return false;
  }
  // Case 2: first wait
  if (nodeNum == 0) {
    SK_LOGD("[lock detector] Wait node %s: first node in scope, cannot fuse", curNode.Format().c_str());
    deadlockReason_ = DeadlockFailReason::FIRST_WAIT;
    return false;
  }
  // Case 3: notify node is in the same SK stream
  if (IsInSKStream(*notifyNode)) {
    if (!CheckNotifyInSKStream(curNode, *notifyNode)) {
      return false;
    }
  } else if (!HasIntersection(skStreamIds, notifyNode->GetScopeStreamIds())) {
    // A notify outside the SK streams needs resource and dependency checks.
    // Case 5: notify node has core resource requirement
    if (notifyNode->GetCubeNum() > 0 || notifyNode->GetVecNum() > 0) {
      bool canFuse = HasEnoughCores(notifyNode, false);
      SK_LOGD("[lock detector] Wait node %s: notify %s has cores, canFuse=%d", curNode.Format().c_str(),
              notifyNode->Format().c_str(), canFuse);
      if (canFuse) {
        tempVisitedNodes.emplace_back(notifyNode->GetNodeId());
      } else {
        deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
        return false;
      }
    }

    // Case 6: notify node is in different stream, check for deadlock
    bool hasDeadlock = HasDeadlock(notifyNode);
    SK_LOGD("[lock detector] Wait node %s: notify %s HasDeadlock=%d", curNode.Format().c_str(),
            notifyNode->Format().c_str(), hasDeadlock);
    if (hasDeadlock) {
      return false;
    }
  }

  std::unordered_set<uint32_t> predecessorStreamIds;
  for (const auto nodeId : tempVisitedNodes) {
    const auto *node = graph_->GetNodeById(nodeId);
    if (node != nullptr) {
      predecessorStreamIds.insert(node->GetStreamIdxInGraph());
    }
  }
  if (!CheckNotifyWaitScopeCombinations(curNode, *notifyNode, superKernelCubeNum, superKernelVecNum, depOpCubeNum,
                                        depOpVecNum, predecessorStreamIds)) {
    deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
    return false;
  }
  fusedNotifyWaitInfos_.emplace(notifyId, FusedNotifyWaitInfo{curNode.GetNodeId(), std::move(predecessorStreamIds)});
  return true;
}

bool LockDetector::CheckNotifyInSKStream(SuperKernelBaseNode &curNode, SuperKernelBaseNode &notifyNode) {
  if (IsAfterSKRange(notifyNode)) {
    SK_LOGE("[lock detector] Wait node %s: notify %s is after SK range, cannot fuse", curNode.Format().c_str(),
            notifyNode.Format().c_str());
    deadlockReason_ = DeadlockFailReason::NOTIFY_AFTER_SK_RANGE;
    return false;
  }
  SK_LOGD("[lock detector] Wait node %s: notify %s is before SK range, can fuse", curNode.Format().c_str(),
          notifyNode.Format().c_str());
  return true;
}

bool LockDetector::ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const {
  if (curNode.GetNodeType() != SkNodeType::NODE_WAIT) {
    return false;
  }

  const auto &syncInfos = curNode.GetNodeInfos().syncInfos;
  const auto *option = opts_ == nullptr ? nullptr
                                        : static_cast<const AggressiveOptStrategiesOption *>(
                                              opts_->GetOption(aclskOptionType::AGGRESSIVE_OPT_STRATEGIES));
  const uint32_t valueBreakerBypass =
      option == nullptr ? ACLSK_VALUE_BREAKER_BYPASS_NONE : option->GetValue().valueBreakerBypass;
  // Only 0b10 keeps unpaired value waits alive through deadlock refine.
  return syncInfos.addrValue != nullptr && curNode.GetCorrespondingNotifyNodeId() == INVALID_TASK_ID &&
         (valueBreakerBypass & ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT) != 0;
}

bool LockDetector::GetFusibleStatus(SuperKernelBaseNode &curNode) {
  if (curNode.GetNodeType() == SkNodeType::NODE_NOTIFY) {
    if (curNode.GetCubeNum() == 0 && curNode.GetVecNum() == 0) {
      SK_LOGD("[lock detector] Notify node %s: not needed core resource, can fuse", curNode.Format().c_str());
      return true;
    } else {
      SK_LOGE("[lock detector] Notify node %s: in SK range with coreNum>0 (cube %u, vec %u), which is not allowed",
              curNode.Format().c_str(), curNode.GetCubeNum(), curNode.GetVecNum());
      deadlockReason_ = DeadlockFailReason::NOTIFY_INVALID;
      return false;
    }
  } else if (curNode.GetNodeType() == SkNodeType::NODE_WAIT) {
    if (ShouldBypassValueWaitDeadlock(curNode)) {
      SK_LOGI("[lock detector] Wait node %s bypassed deadlock detection by value breaker policy",
              curNode.Format().c_str());
      return true;
    }
    tempVisitedNodes.clear();
    bool canFuse = GetWaitNodeFusibleStatus(curNode);
    if (canFuse) {
      nodes.insert(nodes.end(), tempVisitedNodes.begin(), tempVisitedNodes.end());
    } else {
      RollbackVisitedState(tempVisitedNodes);
    }
    return canFuse;
  } else if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL) {
    if (curNode.IsScopeNode()) {
      SK_LOGD("[lock detector] Scope marker %s: no core resource, can fuse", curNode.Format().c_str());
      return true;
    }
    uint32_t cubeNum = curNode.GetCubeNum();
    uint32_t vecNum = curNode.GetVecNum();
    SK_LOGD("[lock detector] Kernel node %s: coreNum={%u, %u}, superKernelCubeNum=%u, superKernelVecNum=%u",
            curNode.Format().c_str(), cubeNum, vecNum, superKernelCubeNum, superKernelVecNum);
    return HasEnoughCores(&curNode, true);
  } else if (curNode.GetNodeType() == SkNodeType::NODE_RESET) {
    SK_LOGD("[lock detector] Reset node %s: no core resource, can fuse", curNode.Format().c_str());
    return true;
  } else if (curNode.GetNodeType() == SkNodeType::NODE_DEFAULT) {
    SK_LOGD("[lock detector] Default node %s: no core resource, can fuse", curNode.Format().c_str());
    return true;
  } else {
    SK_LOGW("[lock detector] Node %s: unsupported taskType=%u", curNode.Format().c_str(), curNode.GetNodeType());
    deadlockReason_ = DeadlockFailReason::NO_SUPPORT_NODE;
    return false;
  }
}

bool LockDetector::IsFusible(SuperKernelBaseNode &curNode) {
  // If node already visited (already checked or fused), return true directly without modifying state
  if (curNode.IsVisited()) {
    SK_LOGD("[lock detector] Node %s: already visited, can fuse", curNode.Format().c_str());
    return true;
  }

  SK_LOGI("[lock detector] IsFusible: Checking node %s, current state: nodeNum=%u, kernelNodeNum=%u",
          curNode.Format().c_str(), nodeNum, kernelNodeNum);
  deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;  // Reset before checking
  bool canFuse = GetFusibleStatus(curNode);
  // Only modify state if node can be fused
  if (canFuse) {
    skStreamIds.insert(curNode.GetStreamIdxInGraph());
    UpdateSKRangeInStream(curNode);
    nodeNum++;
    curNode.SetVisited(true);
    nodes.emplace_back(curNode.GetNodeId());

    if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL && !curNode.IsScopeNode()) {
      kernelNodeNum++;
    }
    SK_LOGD(
        "[lock detector] fused nodeId=%s, nodeType=%u, nodeNum=%u, SuperKernelCubeNum=%u, SuperKernelVecNum=%u, "
        "depOpCubeNum=%u, depOpVecNum=%u",
        curNode.Format().c_str(), curNode.GetNodeType(), nodeNum, superKernelCubeNum, superKernelVecNum, depOpCubeNum,
        depOpVecNum);
  } else {
    SK_LOGI("[lock detector] Node %s: cannot be fused", curNode.Format().c_str());
    // If deadlock was detected, set the failure reason with detail
    if (deadlockReason_ != DeadlockFailReason::NOT_FIND_DEADLOCK) {
      curNode.SetFusionFailReason(deadlockReason_);
    }
  }

  return canFuse;
}

void LockDetector::SetNotifyNodesExpandNumForScope(SuperKernelScopeInfo &scope, const ScopeCoreInfo &scopeCoreInfo) {
  const uint32_t maxExpandVecNum = scopeCoreInfo.GetVectorNum();
  const uint32_t maxExpandCubeNum = scopeCoreInfo.GetCubeNum();
  std::vector<SuperKernelBaseNode *> notifyNodes;
  std::unordered_set<uint32_t> scopeStreams;
  // Collect notify nodes and stream information using the current deadlock-check candidate.
  for (const auto *node : scope.GetNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
      notifyNodes.push_back(const_cast<SuperKernelBaseNode *>(node));
    }
    scopeStreams.insert(node->GetStreamIdxInGraph());
  }

  // Set expand numbers for all notify nodes
  for (auto *notifyNode : notifyNodes) {
    notifyNode->SetNotifyExpandVecNum(maxExpandVecNum);
    notifyNode->SetNotifyExpandCubeNum(maxExpandCubeNum);
    SK_LOGI("[lock detector] Set Notify node %lu expandVecNum=%u, expandCubeNum=%u", notifyNode->GetNodeId(),
            maxExpandVecNum, maxExpandCubeNum);
  }
  // Set expand stream for all node
  for (auto *node : scope.GetNodes()) {
    node->SetScopeStreamIds(scopeStreams);
  }
}

void LockDetector::ResetNotifyExpandNumForScope(SuperKernelScopeInfo &scope) {
  for (auto *node : scope.GetNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
      node->SetNotifyExpandVecNum(0);
      node->SetNotifyExpandCubeNum(0);
      SK_LOGD("[lock detector] Reset Notify node %lu expandVecNum=0, expandCubeNum=0", node->GetNodeId());
    }
    node->SetScopeStreamIds({});
  }
}
