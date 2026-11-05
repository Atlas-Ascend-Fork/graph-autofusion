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

std::string ResourceConflictInfo::Format() const {
  if (parallelNodeIds.empty() && parallelScopeIds.empty()) {
    return "";
  }
  const auto formatIds = [](const auto &ids) {
    auto sortedIds = ids;
    std::sort(sortedIds.begin(), sortedIds.end());
    sortedIds.erase(std::unique(sortedIds.begin(), sortedIds.end()), sortedIds.end());
    std::string result;
    for (const auto id : sortedIds) {
      if (!result.empty()) {
        result += ",";
      }
      result += std::to_string(id);
    }
    return result;
  };
  return "parallelNodes=[" + formatIds(parallelNodeIds) + "], parallelScopes=[" + formatIds(parallelScopeIds) + "]";
}

// LockDetector: device resources, scope index, and shared analyzer entry points.
void LockDetector::Init(SuperKernelGraph &graph) {
  SK_LOGD("[lock detector] LockDetector::Init: Initializing lock detector");
  nodes.clear();
  candidateScopeNodes_.clear();
  activeConcurrentRanges_.clear();
  depOpCubeNum = 0;
  depOpVecNum = 0;
  depOpCubeSourceNodeId_ = INVALID_TASK_ID;
  depOpVectorSourceNodeId_ = INVALID_TASK_ID;
  resourceConflictInfo_ = {};
  superKernelCubeNum = 0;
  superKernelVecNum = 0;
  currentScopeId_ = INVALID_SCOPE_ID;
  scopeCoreInfoChanged_ = false;
  nodeNum = 0;
  kernelNodeNum = 0;
  skStreamIds.clear();
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
  return waitResources_.HasEnoughDeviceCores(skCubeNum, skVecNum, depCubeNum, depVecNum);
}

std::unordered_map<uint32_t, LockDetector::ConcurrentNodeRange> LockDetector::CalculateScopeConcurrentRanges(
    const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId) const {
  return concurrentRanges_.CalculateScopeConcurrentRanges(candidate, candidateScopeId);
}

bool LockDetector::CanRunTogether(uint16_t firstScopeId, uint16_t secondScopeId,
                                  const ScopeConcurrentRanges &ranges) const {
  return concurrentRanges_.CanRunTogether(firstScopeId, secondScopeId, ranges);
}

bool LockDetector::CheckWaitResourceBranches(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                             const CoreDemand &rootDemand,
                                             const std::vector<SuperKernelBaseNode *> &rootNodes,
                                             const std::unordered_set<uint16_t> &initialResidents,
                                             const ScopeConcurrentRanges *precomputedRanges,
                                             ResourceConflictInfo *conflictInfo) {
  return waitResources_.CheckWaitResourceBranches(waitNode, rootScopeId, rootDemand, rootNodes, initialResidents,
                                                  precomputedRanges, conflictInfo);
}

bool LockDetector::ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const {
  return waitResources_.ShouldBypassValueWaitDeadlock(curNode);
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
  scopeInfos_[scope.GetScopeId()] = ScopeRuntimeInfo::FromScope(scope);
}

void LockDetector::RemoveScopeInfo(uint16_t scopeId) {
  scopeInfos_.erase(scopeId);
}

ScopeRuntimeInfo ScopeRuntimeInfo::FromScope(const SuperKernelScopeInfo &scope) {
  ScopeRuntimeInfo scopeInfo;
  scopeInfo.coreInfo = scope.GetScopeCoreInfo();
  scopeInfo.memberNodes = scope.GetNodes();
  scopeInfo.hasWait = std::any_of(scopeInfo.memberNodes.begin(), scopeInfo.memberNodes.end(),
                                  [](const auto *node) { return node->GetNodeType() == SkNodeType::NODE_WAIT; });
  for (const auto &streamInfo : scope.GetScopeStreamInfos()) {
    scopeInfo.streamIds.insert(streamInfo.streamIdx);
    scopeInfo.headNodeIds.emplace(streamInfo.streamIdx, streamInfo.headNodeIdx);
    scopeInfo.tailNodeIds.emplace(streamInfo.streamIdx, streamInfo.tailNodeIdx);
  }
  return scopeInfo;
}

bool LockDetector::IsInSKStream(const SuperKernelBaseNode &node) const {
  return std::find(skStreamIds.begin(), skStreamIds.end(), node.GetStreamIdxInGraph()) != skStreamIds.end();
}

// ScopeConcurrentRangeAnalyzer: calculate per-stream overlap windows, then traverse dependency boundaries.
std::unordered_map<uint32_t, ScopeConcurrentRangeAnalyzer::ConcurrentNodeRange>
ScopeConcurrentRangeAnalyzer::CalculateScopeConcurrentRanges(const std::vector<SuperKernelBaseNode *> &candidate,
                                                             uint16_t candidateScopeId) const {
  ConcurrentRangeState rangeContext;
  InitializeConcurrentRangeState(candidate, candidateScopeId, rangeContext);
  CalculateLeftConcurrentBoundary(rangeContext);
  CalculateRightConcurrentBoundary(rangeContext);
  return rangeContext.concurrentRangesByStream;
}

void ScopeConcurrentRangeAnalyzer::CalculateLeftConcurrentBoundary(ConcurrentRangeState &rangeContext) const {
  TraverseConcurrentBoundary(rangeContext, true);
}

void ScopeConcurrentRangeAnalyzer::CalculateRightConcurrentBoundary(ConcurrentRangeState &rangeContext) const {
  TraverseConcurrentBoundary(rangeContext, false);
}

// Resolve scope boundaries and initialize the candidate's per-stream windows.
SuperKernelBaseNode *ScopeConcurrentRangeAnalyzer::GetScopeBoundary(const ConcurrentRangeState &rangeContext,
                                                                    uint32_t scopeId, uint32_t streamId,
                                                                    bool first) const {
  if (scopeId == CANDIDATE_CONCURRENT_SCOPE) {
    const auto boundaryIt = rangeContext.candidateBoundaries.find(streamId);
    if (boundaryIt == rangeContext.candidateBoundaries.end()) {
      return nullptr;
    }
    const auto nodeId = first ? boundaryIt->second.firstNodeId : boundaryIt->second.lastNodeId;
    return graph_->GetNodeById(nodeId);
  }
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeIt == scopeInfos_.end()) {
    return nullptr;
  }
  const auto &nodeIds = first ? scopeIt->second.headNodeIds : scopeIt->second.tailNodeIds;
  const auto nodeIt = nodeIds.find(streamId);
  return nodeIt == nodeIds.end() ? nullptr : graph_->GetNodeById(nodeIt->second);
}

uint32_t ScopeConcurrentRangeAnalyzer::GetTrackedScopeId(const ConcurrentRangeState &rangeContext,
                                                         const SuperKernelBaseNode &node) const {
  if (std::any_of(rangeContext.candidateNodes->begin(), rangeContext.candidateNodes->end(),
                  [&](const auto *member) { return member->GetNodeId() == node.GetNodeId(); })) {
    return CANDIDATE_CONCURRENT_SCOPE;
  }
  const auto scopeIt = scopeInfos_.find(node.GetScopeId());
  if (scopeIt != scopeInfos_.end() && scopeIt->first != rangeContext.candidateScopeId) {
    return scopeIt->first;
  }
  return INVALID_SCOPE_ID;
}

void ScopeConcurrentRangeAnalyzer::InitializeConcurrentRangeState(const std::vector<SuperKernelBaseNode *> &candidate,
                                                                  uint16_t candidateScopeId,
                                                                  ConcurrentRangeState &rangeContext) const {
  rangeContext.candidateNodes = &candidate;
  rangeContext.candidateScopeId = candidateScopeId;
  for (const auto *node : candidate) {
    auto &range = rangeContext.candidateBoundaries[node->GetStreamIdxInGraph()];
    if (range.firstNodeId == INVALID_TASK_ID ||
        node->GetNodeIdxInStream() < graph_->GetNodeById(range.firstNodeId)->GetNodeIdxInStream()) {
      range.firstNodeId = node->GetNodeId();
    }
    if (range.lastNodeId == INVALID_TASK_ID ||
        node->GetNodeIdxInStream() > graph_->GetNodeById(range.lastNodeId)->GetNodeIdxInStream()) {
      range.lastNodeId = node->GetNodeId();
    }
  }
  rangeContext.concurrentRangesByStream = rangeContext.candidateBoundaries;
  const auto &headNodes = graph_->GetHeadNodes();
  const auto &tailNodes = graph_->GetTailNodes();
  for (uint32_t streamId = 0; streamId < headNodes.size(); ++streamId) {
    if (headNodes[streamId] != INVALID_TASK_ID && rangeContext.concurrentRangesByStream.count(streamId) == 0) {
      rangeContext.concurrentRangesByStream.emplace(streamId,
                                                    ConcurrentNodeRange{headNodes[streamId], tailNodes[streamId]});
    }
  }
}

// Tighten windows when traversal crosses a stream, event, or multi-stream scope.
void ScopeConcurrentRangeAnalyzer::TightenConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t streamId,
                                                             const SuperKernelBaseNode *boundary, bool left) const {
  if (GetScopeBoundary(rangeContext, CANDIDATE_CONCURRENT_SCOPE, streamId, true) != nullptr) {
    return;
  }
  const auto rangeIt = rangeContext.concurrentRangesByStream.find(streamId);
  if (rangeIt == rangeContext.concurrentRangesByStream.end() || rangeIt->second.firstNodeId == INVALID_TASK_ID) {
    return;
  }
  auto &range = rangeIt->second;
  if (boundary == nullptr) {
    range = {};
    return;
  }
  const auto *first = graph_->GetNodeById(range.firstNodeId);
  const auto *last = graph_->GetNodeById(range.lastNodeId);
  if (left && boundary->GetNodeIdxInStream() > first->GetNodeIdxInStream()) {
    range.firstNodeId = boundary->GetNodeId();
  } else if (!left && boundary->GetNodeIdxInStream() < last->GetNodeIdxInStream()) {
    range.lastNodeId = boundary->GetNodeId();
  }
  if ((left && boundary->GetNodeIdxInStream() > last->GetNodeIdxInStream()) ||
      (!left && boundary->GetNodeIdxInStream() < first->GetNodeIdxInStream())) {
    range = {};
  }
}

void ScopeConcurrentRangeAnalyzer::TightenScopeConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t scopeId,
                                                                  uint32_t entryStreamId, bool left,
                                                                  bool eventBoundary) const {
  if (scopeId == CANDIDATE_CONCURRENT_SCOPE) {
    return;
  }
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeIt == scopeInfos_.end()) {
    return;
  }
  for (const auto streamId : scopeIt->second.streamIds) {
    if (!eventBoundary && streamId == entryStreamId) {
      continue;
    }
    const auto *boundary = GetScopeBoundary(rangeContext, scopeId, streamId, eventBoundary ? left : !left);
    if (boundary == nullptr) {
      continue;
    }
    const auto adjacentId = left != eventBoundary ? boundary->GetNextNodeId() : boundary->GetPreNodeId();
    if (eventBoundary && adjacentId == INVALID_TASK_ID) {
      continue;
    }
    boundary = adjacentId == INVALID_TASK_ID ? nullptr : graph_->GetNodeById(adjacentId);
    TightenConcurrentBoundary(rangeContext, streamId, boundary, left);
  }
}

void ScopeConcurrentRangeAnalyzer::UpdateConcurrentBoundary(ConcurrentRangeState &rangeContext,
                                                            const ConcurrentVertex &vertex, bool left) const {
  if (vertex.kind == ConcurrentVertex::Kind::SCOPE) {
    return;
  }
  const auto *node = graph_->GetNodeById(vertex.vertexNodeId);
  if (node == nullptr) {
    return;
  }
  const auto scopeId = GetTrackedScopeId(rangeContext, *node);
  if (vertex.kind == ConcurrentVertex::Kind::EVENT_NODE) {
    if (scopeId != INVALID_SCOPE_ID) {
      TightenScopeConcurrentBoundary(rangeContext, scopeId, node->GetStreamIdxInGraph(), left, true);
    } else {
      const auto adjacentId = left ? node->GetNextNodeId() : node->GetPreNodeId();
      const auto *boundary = adjacentId == INVALID_TASK_ID ? nullptr : graph_->GetNodeById(adjacentId);
      TightenConcurrentBoundary(rangeContext, node->GetStreamIdxInGraph(), boundary, left);
    }
    return;
  }
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeIt != scopeInfos_.end() && scopeIt->second.streamIds.size() > 1 && vertex.activeScopeId != scopeId) {
    TightenScopeConcurrentBoundary(rangeContext, scopeId, node->GetStreamIdxInGraph(), left, false);
  }
}

// Expand traversal vertices in stream order and across Notify/Wait edges.
void ScopeConcurrentRangeAnalyzer::QueueConcurrentBoundarySeeds(const ConcurrentRangeState &rangeContext, bool left,
                                                                std::deque<ConcurrentVertex> &pending) const {
  for (const auto *member : *rangeContext.candidateNodes) {
    const auto *boundary =
        GetScopeBoundary(rangeContext, CANDIDATE_CONCURRENT_SCOPE, member->GetStreamIdxInGraph(), left);
    if (boundary->GetNodeId() != member->GetNodeId()) {
      continue;
    }
    const auto adjacentId = left ? boundary->GetPreNodeId() : boundary->GetNextNodeId();
    if (adjacentId != INVALID_TASK_ID) {
      const auto *adjacent = graph_->GetNodeById(adjacentId);
      if (adjacent != nullptr) {
        pending.push_back({ConcurrentVertex::Kind::NODE, INVALID_SCOPE_ID, adjacent->GetNodeId()});
      }
    }
  }
}

void ScopeConcurrentRangeAnalyzer::AppendScopeVertices(const ConcurrentRangeState &rangeContext,
                                                       const ConcurrentVertex &vertex, bool left,
                                                       std::deque<ConcurrentVertex> &pending) const {
  const auto scopeIt = scopeInfos_.find(vertex.activeScopeId);
  if (scopeIt == scopeInfos_.end()) {
    return;
  }
  for (const auto streamId : scopeIt->second.streamIds) {
    const auto *boundary = GetScopeBoundary(rangeContext, vertex.activeScopeId, streamId, !left);
    if (boundary != nullptr) {
      pending.push_back({ConcurrentVertex::Kind::NODE, vertex.activeScopeId, boundary->GetNodeId()});
    }
  }
}

void ScopeConcurrentRangeAnalyzer::AppendConcurrentStreamVertex(const ConcurrentRangeState &rangeContext,
                                                                const SuperKernelBaseNode &node, bool left,
                                                                const ConcurrentVertex &vertex,
                                                                std::deque<ConcurrentVertex> &pending) const {
  const auto scopeId = GetTrackedScopeId(rangeContext, node);
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeIt != scopeInfos_.end() && scopeIt->second.streamIds.size() > 1 && vertex.activeScopeId != scopeId) {
    pending.push_back({ConcurrentVertex::Kind::SCOPE, scopeId, INVALID_TASK_ID});
  }
  const auto adjacentId = left ? node.GetPreNodeId() : node.GetNextNodeId();
  if (adjacentId == INVALID_TASK_ID) {
    return;
  }
  const auto *adjacent = graph_->GetNodeById(adjacentId);
  if (adjacent == nullptr) {
    return;
  }
  const auto nextScopeId = GetTrackedScopeId(rangeContext, *adjacent) == scopeId ? scopeId : INVALID_SCOPE_ID;
  pending.push_back({ConcurrentVertex::Kind::NODE, nextScopeId, adjacentId});
}

void ScopeConcurrentRangeAnalyzer::AppendForwardVertices(const ConcurrentRangeState &rangeContext,
                                                         const ConcurrentVertex &vertex,
                                                         std::deque<ConcurrentVertex> &pending) const {
  if (vertex.kind == ConcurrentVertex::Kind::SCOPE) {
    AppendScopeVertices(rangeContext, vertex, false, pending);
    return;
  }
  const auto *node = graph_->GetNodeById(vertex.vertexNodeId);
  if (node == nullptr) {
    return;
  }
  if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
    for (const auto waitId : node->GetCorrespondingWaitNodeIds()) {
      if (waitId != INVALID_TASK_ID) {
        pending.push_back({ConcurrentVertex::Kind::EVENT_NODE, INVALID_SCOPE_ID, waitId});
      }
    }
  }
  AppendConcurrentStreamVertex(rangeContext, *node, false, vertex, pending);
}

void ScopeConcurrentRangeAnalyzer::AppendReverseVertices(const ConcurrentRangeState &rangeContext,
                                                         const ConcurrentVertex &vertex,
                                                         std::deque<ConcurrentVertex> &pending) const {
  if (vertex.kind == ConcurrentVertex::Kind::SCOPE) {
    AppendScopeVertices(rangeContext, vertex, true, pending);
    return;
  }
  const auto *node = graph_->GetNodeById(vertex.vertexNodeId);
  if (node == nullptr) {
    return;
  }
  if (node->GetNodeType() == SkNodeType::NODE_WAIT && node->GetCorrespondingNotifyNodeId() != INVALID_TASK_ID) {
    pending.push_back({ConcurrentVertex::Kind::EVENT_NODE, INVALID_SCOPE_ID, node->GetCorrespondingNotifyNodeId()});
  }
  AppendConcurrentStreamVertex(rangeContext, *node, true, vertex, pending);
}

void ScopeConcurrentRangeAnalyzer::TraverseConcurrentBoundary(ConcurrentRangeState &rangeContext, bool left) const {
  std::deque<ConcurrentVertex> pending;
  std::set<ConcurrentVertex> visited;
  QueueConcurrentBoundarySeeds(rangeContext, left, pending);
  while (!pending.empty()) {
    const auto vertex = pending.front();
    pending.pop_front();
    if (!visited.insert(vertex).second) {
      continue;
    }
    UpdateConcurrentBoundary(rangeContext, vertex, left);
    if (left) {
      AppendReverseVertices(rangeContext, vertex, pending);
    } else {
      AppendForwardVertices(rangeContext, vertex, pending);
    }
  }
}

// ScopeConcurrentRangeAnalyzer: compare the calculated windows between scopes.
bool ScopeConcurrentRangeAnalyzer::IsBeforeConcurrentRange(
    const SuperKernelBaseNode &node, const std::unordered_map<uint32_t, ConcurrentNodeRange> &range) const {
  const auto it = range.find(node.GetStreamIdxInGraph());
  if (it == range.end()) {
    return false;
  }
  if (it->second.firstNodeId == INVALID_TASK_ID) {
    return true;
  }
  const auto *first = graph_->GetNodeById(it->second.firstNodeId);
  return first != nullptr && node.GetNodeIdxInStream() < first->GetNodeIdxInStream();
}

bool ScopeConcurrentRangeAnalyzer::CanRunTogether(uint16_t firstScopeId, uint16_t secondScopeId,
                                                  const ScopeConcurrentRanges &ranges) const {
  const auto firstIt = scopeInfos_.find(firstScopeId);
  const auto secondIt = scopeInfos_.find(secondScopeId);
  if (firstIt == scopeInfos_.end() || secondIt == scopeInfos_.end() ||
      HasIntersection(firstIt->second.streamIds, secondIt->second.streamIds)) {
    return false;
  }
  const auto overlaps = [&](uint16_t rootId, const ScopeRuntimeInfo &other) {
    const auto rangeIt = ranges.find(rootId);
    if (rangeIt == ranges.end()) {
      return true;
    }
    for (const auto streamId : other.streamIds) {
      const auto windowIt = rangeIt->second.find(streamId);
      if (windowIt == rangeIt->second.end() || windowIt->second.firstNodeId == INVALID_TASK_ID) {
        return false;
      }
      const auto *first = graph_->GetNodeById(windowIt->second.firstNodeId);
      const auto *last = graph_->GetNodeById(windowIt->second.lastNodeId);
      const auto *head = graph_->GetNodeById(other.headNodeIds.at(streamId));
      const auto *tail = graph_->GetNodeById(other.tailNodeIds.at(streamId));
      if (first == nullptr || last == nullptr || head == nullptr || tail == nullptr ||
          tail->GetNodeIdxInStream() < first->GetNodeIdxInStream() ||
          head->GetNodeIdxInStream() > last->GetNodeIdxInStream()) {
        return false;
      }
    }
    return true;
  };
  return overlaps(firstScopeId, secondIt->second) && overlaps(secondScopeId, firstIt->second);
}

bool ScopeConcurrentRangeAnalyzer::HasIntersection(const std::unordered_set<uint32_t> &lhsStreams,
                                                   const std::unordered_set<uint32_t> &rhsStreams) const {
  for (const auto &streamId : lhsStreams) {
    if (rhsStreams.count(streamId) > 0) {
      return true;
    }
  }
  return false;
}

// WaitResourceDependencyAnalyzer: search Notify dependencies and evaluate each resource branch.
bool WaitResourceDependencyAnalyzer::CheckWaitResourceBranches(const SuperKernelBaseNode &waitNode,
                                                               uint16_t rootScopeId, const CoreDemand &rootDemand,
                                                               const std::vector<SuperKernelBaseNode *> &rootNodes,
                                                               const std::unordered_set<uint16_t> &initialResidents,
                                                               const ScopeConcurrentRanges *precomputedRanges,
                                                               ResourceConflictInfo *conflictInfo) {
  if (conflictInfo != nullptr) {
    *conflictInfo = {};
  }
  ScopeConcurrentRanges calculatedRanges;
  ResourceSearchState initial;
  PrepareResourceSearch(waitNode, rootScopeId, rootDemand, rootNodes, initialResidents, calculatedRanges, initial,
                        precomputedRanges);
  const auto &ranges = precomputedRanges == nullptr ? calculatedRanges : *precomputedRanges;
  if (!HasEnoughDeviceCores(initial.residentDemand.cubeCoreNum, initial.residentDemand.vectorCoreNum, 0, 0)) {
    if (conflictInfo != nullptr) {
      const bool cubeRisk = deviceRealCubeNum < 0 || initial.residentDemand.cubeCoreNum > deviceRealCubeNum;
      const bool vectorRisk = deviceRealVecNum < 0 || initial.residentDemand.vectorCoreNum > deviceRealVecNum;
      for (const auto scopeId : initial.residentScopes) {
        const auto scopeIt = scopeInfos_.find(scopeId);
        if (scopeId != rootScopeId && scopeIt != scopeInfos_.end() &&
            ((cubeRisk && scopeIt->second.coreInfo.GetCubeNum() != 0) ||
             (vectorRisk && scopeIt->second.coreInfo.GetVectorNum() != 0))) {
          conflictInfo->parallelScopeIds.push_back(scopeId);
        }
      }
    }
    SK_LOGI("[ParallelResource] Wait %lu: resident scopes require cube=%lu, vec=%lu; device has cube=%ld, vec=%ld",
            waitNode.GetNodeId(), initial.residentDemand.cubeCoreNum, initial.residentDemand.vectorCoreNum,
            deviceRealCubeNum, deviceRealVecNum);
    return false;
  }

  std::vector<ResourceSearchState> branches;
  branches.push_back(std::move(initial));
  while (!branches.empty()) {
    ResourceSearchState state = std::move(branches.back());
    branches.pop_back();
    while (!state.pendingNodeIds.empty()) {
      const auto nodeId = state.pendingNodeIds.back();
      state.pendingNodeIds.pop_back();
      if (!ProcessResourceSearchNode(nodeId, rootScopeId, rootNodes, ranges, state, branches)) {
        SK_LOGE("[ParallelResource] Wait %lu: failed to search dependency node %lu", waitNode.GetNodeId(), nodeId);
        return false;
      }
    }
    if (ResourceSearchHasRisk(state, rootScopeId, conflictInfo)) {
      SK_LOGI("[ParallelResource] Wait %lu in scope %u has resource risk", waitNode.GetNodeId(), rootScopeId);
      return false;
    }
  }
  return true;
}

// Expand resident and external scope branches, then accumulate their core demand.
void WaitResourceDependencyAnalyzer::QueueScopeWaitNotifies(uint16_t scopeId, ResourceSearchState &state) const {
  if (!state.expandedWaitScopes.insert(scopeId).second) {
    return;
  }
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeIt == scopeInfos_.end()) {
    return;
  }
  for (const auto *node : scopeIt->second.memberNodes) {
    if (node->GetNodeType() == SkNodeType::NODE_WAIT && !ShouldBypassValueWaitDeadlock(*node)) {
      const auto notifyId = node->GetCorrespondingNotifyNodeId();
      const auto *notify = graph_->GetNodeById(notifyId);
      if (notify == nullptr || notify->GetScopeId() != scopeId) {
        state.pendingNodeIds.push_back(notifyId);
      }
    }
  }
}

bool WaitResourceDependencyAnalyzer::ResourceSearchHasRisk(const ResourceSearchState &state, uint16_t rootScopeId,
                                                           ResourceConflictInfo *conflictInfo) const {
  uint64_t externalCube = 0;
  uint64_t externalVec = 0;
  for (const auto &[streamId, demand] : state.externalByStream) {
    externalCube = std::max(externalCube, demand.cubeCoreNum);
    externalVec = std::max(externalVec, demand.vectorCoreNum);
  }
  for (const auto &[scopeId, demand] : state.externalScopes) {
    externalCube = std::max(externalCube, demand.cubeCoreNum);
    externalVec = std::max(externalVec, demand.vectorCoreNum);
  }
  const bool hasRisk = !HasEnoughDeviceCores(state.residentDemand.cubeCoreNum, state.residentDemand.vectorCoreNum,
                                             externalCube, externalVec);
  if (hasRisk) {
    if (conflictInfo != nullptr) {
      const bool cubeRisk = deviceRealCubeNum < 0 ||
                            state.residentDemand.cubeCoreNum + externalCube > static_cast<uint64_t>(deviceRealCubeNum);
      const bool vectorRisk = deviceRealVecNum < 0 || state.residentDemand.vectorCoreNum + externalVec >
                                                          static_cast<uint64_t>(deviceRealVecNum);
      for (const auto scopeId : state.residentScopes) {
        const auto scopeIt = scopeInfos_.find(scopeId);
        if (scopeId != rootScopeId && scopeIt != scopeInfos_.end() &&
            ((cubeRisk && scopeIt->second.coreInfo.GetCubeNum() != 0) ||
             (vectorRisk && scopeIt->second.coreInfo.GetVectorNum() != 0))) {
          conflictInfo->parallelScopeIds.push_back(scopeId);
        }
      }
      for (const auto &[streamId, demand] : state.externalByStream) {
        if (cubeRisk && externalCube != 0 && demand.cubeCoreNum == externalCube) {
          conflictInfo->parallelNodeIds.push_back(state.cubeSourceNodeByStream.at(streamId));
        }
        if (vectorRisk && externalVec != 0 && demand.vectorCoreNum == externalVec) {
          conflictInfo->parallelNodeIds.push_back(state.vectorSourceNodeByStream.at(streamId));
        }
      }
      for (const auto &[scopeId, demand] : state.externalScopes) {
        if ((cubeRisk && externalCube != 0 && demand.cubeCoreNum == externalCube) ||
            (vectorRisk && externalVec != 0 && demand.vectorCoreNum == externalVec)) {
          conflictInfo->parallelScopeIds.push_back(scopeId);
        }
      }
    }
    SK_LOGI(
        "[ParallelResource] Resource risk: residentCube=%lu, externalCube=%lu, totalCube=%lu, deviceCube=%ld, "
        "residentVec=%lu, externalVec=%lu, totalVec=%lu, deviceVec=%ld",
        state.residentDemand.cubeCoreNum, externalCube, state.residentDemand.cubeCoreNum + externalCube,
        deviceRealCubeNum, state.residentDemand.vectorCoreNum, externalVec,
        state.residentDemand.vectorCoreNum + externalVec, deviceRealVecNum);
    for (const auto &[streamId, demand] : state.externalByStream) {
      SK_LOGI("[ParallelResource] External stream %u: cube=%lu, vec=%lu", streamId, demand.cubeCoreNum,
              demand.vectorCoreNum);
    }
    for (const auto &[scopeId, demand] : state.externalScopes) {
      SK_LOGI("[ParallelResource] External scope %u: cube=%lu, vec=%lu", scopeId, demand.cubeCoreNum,
              demand.vectorCoreNum);
    }
  }
  return hasRisk;
}

void WaitResourceDependencyAnalyzer::PrepareResourceSearch(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                                           const CoreDemand &rootDemand,
                                                           const std::vector<SuperKernelBaseNode *> &rootNodes,
                                                           const std::unordered_set<uint16_t> &initialResidents,
                                                           ScopeConcurrentRanges &ranges, ResourceSearchState &state,
                                                           const ScopeConcurrentRanges *precomputedRanges) {
  if (precomputedRanges == nullptr) {
    ranges.emplace(rootScopeId, ranges_.CalculateScopeConcurrentRanges(rootNodes, rootScopeId));
    for (const auto &[scopeId, info] : scopeInfos_) {
      if (scopeId != rootScopeId && info.hasWait) {
        ranges.emplace(scopeId, ranges_.CalculateScopeConcurrentRanges(info.memberNodes, scopeId));
      }
    }
  }
  state.pendingNodeIds.push_back(waitNode.GetCorrespondingNotifyNodeId());
  state.residentScopes = initialResidents;
  state.residentScopes.insert(rootScopeId);
  state.residentDemand = rootDemand;
  for (const auto scopeId : state.residentScopes) {
    const auto scopeIt = scopeInfos_.find(scopeId);
    if (scopeId != rootScopeId && scopeIt != scopeInfos_.end()) {
      state.residentDemand.cubeCoreNum += scopeIt->second.coreInfo.GetCubeNum();
      state.residentDemand.vectorCoreNum += scopeIt->second.coreInfo.GetVectorNum();
    }
    if (scopeId == rootScopeId) {
      for (const auto *member : rootNodes) {
        state.occupiedStreams.emplace(member->GetStreamIdxInGraph(), scopeId);
      }
    } else if (scopeIt != scopeInfos_.end()) {
      for (const auto streamId : scopeIt->second.streamIds) {
        state.occupiedStreams.emplace(streamId, scopeId);
      }
    }
  }
}

bool WaitResourceDependencyAnalyzer::CanStartWaitScope(uint16_t scopeId, const ScopeRuntimeInfo &scope,
                                                       uint16_t rootScopeId,
                                                       const std::vector<SuperKernelBaseNode *> &rootNodes,
                                                       const ScopeConcurrentRanges &ranges,
                                                       const ResourceSearchState &state) const {
  for (const auto residentId : state.residentScopes) {
    if (residentId != rootScopeId) {
      if (!ranges_.CanRunTogether(residentId, scopeId, ranges)) {
        return false;
      }
      continue;
    }
    for (const auto *rootNode : rootNodes) {
      if (scope.streamIds.count(rootNode->GetStreamIdxInGraph()) != 0) {
        return false;
      }
    }
    const auto &rootRange = ranges.at(rootScopeId);
    for (const auto scopeStream : scope.streamIds) {
      const auto windowIt = rootRange.find(scopeStream);
      if (windowIt == rootRange.end() || windowIt->second.firstNodeId == INVALID_TASK_ID) {
        return false;
      }
      const auto *first = graph_->GetNodeById(windowIt->second.firstNodeId);
      const auto *last = graph_->GetNodeById(windowIt->second.lastNodeId);
      const auto *head = graph_->GetNodeById(scope.headNodeIds.at(scopeStream));
      const auto *tail = graph_->GetNodeById(scope.tailNodeIds.at(scopeStream));
      if (first == nullptr || last == nullptr || head == nullptr || tail == nullptr ||
          tail->GetNodeIdxInStream() < first->GetNodeIdxInStream() ||
          head->GetNodeIdxInStream() > last->GetNodeIdxInStream()) {
        return false;
      }
    }
  }
  return true;
}

void WaitResourceDependencyAnalyzer::QueueStartedScopeBranch(uint16_t scopeId, const ScopeRuntimeInfo &scope,
                                                             const ResourceSearchState &state,
                                                             std::vector<ResourceSearchState> &branches) const {
  ResourceSearchState started = state;
  started.residentScopes.insert(scopeId);
  started.residentDemand.cubeCoreNum += scope.coreInfo.GetCubeNum();
  started.residentDemand.vectorCoreNum += scope.coreInfo.GetVectorNum();
  started.externalScopes.erase(scopeId);
  for (const auto scopeStream : scope.streamIds) {
    started.occupiedStreams[scopeStream] = scopeId;
    started.externalByStream.erase(scopeStream);
  }
  for (auto it = started.externalScopes.begin(); it != started.externalScopes.end();) {
    const auto otherIt = scopeInfos_.find(it->first);
    if (otherIt != scopeInfos_.end() && ranges_.HasIntersection(scope.streamIds, otherIt->second.streamIds)) {
      it = started.externalScopes.erase(it);
    } else {
      ++it;
    }
  }
  QueueScopeWaitNotifies(scopeId, started);
  branches.push_back(std::move(started));
}

void WaitResourceDependencyAnalyzer::ExpandUnstartedScope(uint16_t scopeId, const ScopeRuntimeInfo &scope,
                                                          ResourceSearchState &state) const {
  state.externalScopes[scopeId] = {scope.coreInfo.GetCubeNum(), scope.coreInfo.GetVectorNum()};
  for (const auto &[headStream, headId] : scope.headNodeIds) {
    const auto *head = graph_->GetNodeById(headId);
    if (head != nullptr && head->GetPreNodeId() != INVALID_TASK_ID) {
      state.pendingNodeIds.push_back(head->GetPreNodeId());
    }
  }
  if (scope.hasWait) {
    QueueScopeWaitNotifies(scopeId, state);
  }
}

bool WaitResourceDependencyAnalyzer::ProcessResourceSearchNode(uint64_t nodeId, uint16_t rootScopeId,
                                                               const std::vector<SuperKernelBaseNode *> &rootNodes,
                                                               const ScopeConcurrentRanges &ranges,
                                                               ResourceSearchState &state,
                                                               std::vector<ResourceSearchState> &branches) {
  if (nodeId == INVALID_TASK_ID) {
    SK_LOGE("[ParallelResource] Invalid node ID in resource search");
    return false;
  }
  const auto *node = graph_->GetNodeById(nodeId);
  if (node == nullptr) {
    SK_LOGE("[ParallelResource] Node %lu not found in resource search", nodeId);
    return false;
  }
  if (!state.visited.insert(nodeId).second || ranges_.IsBeforeConcurrentRange(*node, ranges.at(rootScopeId))) {
    return true;
  }
  const auto streamId = node->GetStreamIdxInGraph();
  const auto scopeId = node->GetScopeId();
  const auto occupied = state.occupiedStreams.find(streamId);
  if (occupied != state.occupiedStreams.end()) {
    if (occupied->second == scopeId && scopeId != rootScopeId) {
      QueueScopeWaitNotifies(scopeId, state);
    }
    return true;
  }
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeId != rootScopeId && scopeIt != scopeInfos_.end()) {
    const auto &scope = scopeIt->second;
    if (scope.hasWait && CanStartWaitScope(scopeId, scope, rootScopeId, rootNodes, ranges, state)) {
      const uint64_t startedCube = state.residentDemand.cubeCoreNum + scope.coreInfo.GetCubeNum();
      const uint64_t startedVec = state.residentDemand.vectorCoreNum + scope.coreInfo.GetVectorNum();
      if (HasEnoughDeviceCores(startedCube, startedVec, 0, 0)) {
        QueueStartedScopeBranch(scopeId, scope, state, branches);
      }
    }
    ExpandUnstartedScope(scopeId, scope, state);
    return true;
  }
  auto &demand = state.externalByStream[streamId];
  if (node->GetCubeNum() > demand.cubeCoreNum) {
    demand.cubeCoreNum = node->GetCubeNum();
    state.cubeSourceNodeByStream[streamId] = nodeId;
  }
  if (node->GetVecNum() > demand.vectorCoreNum) {
    demand.vectorCoreNum = node->GetVecNum();
    state.vectorSourceNodeByStream[streamId] = nodeId;
  }
  if (node->GetPreNodeId() != INVALID_TASK_ID) {
    state.pendingNodeIds.push_back(node->GetPreNodeId());
  }
  if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
    state.pendingNodeIds.push_back(node->GetCorrespondingNotifyNodeId());
  }
  return true;
}

bool WaitResourceDependencyAnalyzer::HasEnoughDeviceCores(uint64_t skCubeNum, uint64_t skVecNum, uint64_t depCubeNum,
                                                          uint64_t depVecNum) const {
  if (deviceRealCubeNum < 0 || deviceRealVecNum < 0 || depCubeNum > static_cast<uint64_t>(deviceRealCubeNum) ||
      depVecNum > static_cast<uint64_t>(deviceRealVecNum)) {
    return false;
  }
  return skCubeNum <= deviceRealCubeNum - depCubeNum && skVecNum <= deviceRealVecNum - depVecNum;
}

bool WaitResourceDependencyAnalyzer::ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const {
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

// LockDetector: traverse a single scope's predecessor dependencies for deadlock.
bool LockDetector::HasDeadlock(SuperKernelBaseNode *curNode, std::unordered_set<uint16_t> &expandedScopeIds) {
  curNode->SetVisited(true);
  tempVisitedNodes.emplace_back(curNode->GetNodeId());
  const auto scopeIt = scopeInfos_.find(curNode->GetScopeId());
  if (curNode->GetScopeId() != currentScopeId_ && scopeIt != scopeInfos_.end()) {
    if (!expandedScopeIds.insert(curNode->GetScopeId()).second) {
      return false;
    }
    return CheckScopeDeadlock(scopeIt->second, expandedScopeIds);
  }
  return HasDeadlockPredecessor(curNode, expandedScopeIds);
}

bool LockDetector::CheckScopeDeadlock(const ScopeRuntimeInfo &scopeInfo,
                                      std::unordered_set<uint16_t> &expandedScopeIds) {
  if (scopeInfo.memberNodes.empty()) {
    return false;
  }
  if (!HasEnoughCores(scopeInfo.memberNodes.front(), false)) {
    deadlockReason_ = DeadlockFailReason::KERNEL_INSUFFICIENT_CORES;
    return true;
  }

  std::unordered_set<uint64_t> memberIds;
  for (const auto *member : scopeInfo.memberNodes) {
    memberIds.insert(member->GetNodeId());
  }
  for (const auto *member : scopeInfo.memberNodes) {
    if (member->GetNodeType() != SkNodeType::NODE_WAIT ||
        memberIds.count(member->GetCorrespondingNotifyNodeId()) != 0) {
      continue;
    }
    auto *notify = graph_->GetNodeById(member->GetCorrespondingNotifyNodeId());
    if (notify == nullptr) {
      deadlockReason_ = DeadlockFailReason::NOTIFY_NOT_IN_GRAPH;
      return true;
    }
    if (IsAfterSKRange(*notify)) {
      deadlockReason_ = DeadlockFailReason::NOTIFY_AFTER_SK_RANGE;
      return true;
    }
    if (notify->GetStreamIdxInGraph() != member->GetStreamIdxInGraph() && !IsInSKStream(*notify) &&
        HasDeadlock(notify, expandedScopeIds)) {
      return true;
    }
  }

  for (const auto &[streamId, headId] : scopeInfo.headNodeIds) {
    auto *head = graph_->GetNodeById(headId);
    if (head == nullptr) {
      continue;
    }
    head->SetVisited(true);
    tempVisitedNodes.emplace_back(headId);
    if (HasDeadlockPredecessor(head, expandedScopeIds)) {
      return true;
    }
  }
  return false;
}

bool LockDetector::HasDeadlockPredecessor(SuperKernelBaseNode *curNode,
                                          std::unordered_set<uint16_t> &expandedScopeIds) {
  if (curNode->GetPreNodeId() == INVALID_TASK_ID) {
    return false;
  }

  const uint64_t preNodeId = curNode->GetPreNodeId();
  SuperKernelBaseNode *preNode = graph_->GetNodeById(preNodeId);
  if (preNode == nullptr) {
    SK_LOGE("[lock detector] HasDeadlock: preNode %lu not found for curNode %lu", preNodeId, curNode->GetNodeId());
    return false;
  }
  if (concurrentRanges_.IsBeforeConcurrentRange(*preNode, activeConcurrentRanges_)) {
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
  if (concurrentRanges_.HasIntersection(preNode->GetScopeStreamIds(), skStreamIds)) {
    return false;
  }

  const auto scopeIt = scopeInfos_.find(preNode->GetScopeId());
  if (preNode->GetScopeId() != currentScopeId_ && scopeIt != scopeInfos_.end()) {
    return HasDeadlock(preNode, expandedScopeIds);
  }

  bool hasDeadlock = true;
  switch (preNode->GetNodeType()) {
    case SkNodeType::NODE_KERNEL:
      hasDeadlock = preNode->IsScopeNode() ? HasDeadlock(preNode, expandedScopeIds)
                                           : CheckKernelNodeDeadlock(preNode, expandedScopeIds);
      break;
    case SkNodeType::NODE_WAIT:
      hasDeadlock = CheckWaitNodeDeadlock(preNode, expandedScopeIds);
      break;
    case SkNodeType::NODE_NOTIFY:
      hasDeadlock = CheckNotifyNodeDeadlock(preNode, expandedScopeIds);
      break;
    case SkNodeType::NODE_DEFAULT:
      hasDeadlock = HasDeadlock(preNode, expandedScopeIds);
      break;
    case SkNodeType::NODE_RESET:
      hasDeadlock = HasDeadlock(preNode, expandedScopeIds);
      break;
    default:
      SK_LOGD("nodeId: %u, unsupported node type %u in HasDeadlock", preNode->GetNodeId(), preNode->GetNodeType());
      deadlockReason_ = DeadlockFailReason::NO_SUPPORT_NODE;
      break;
  }

  return hasDeadlock;
}

bool LockDetector::CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode,
                                           std::unordered_set<uint16_t> &expandedScopeIds) {
  if (!HasEnoughCores(preNode, false)) {
    SK_LOGI("Not enough cores for kernel node, nodeId=%lu, requiredCube=%u, requiredVec=%u", preNode->GetNodeId(),
            preNode->GetCubeNum(), preNode->GetVecNum());
    deadlockReason_ = DeadlockFailReason::KERNEL_INSUFFICIENT_CORES;
    return true;
  }
  if (HasDeadlock(preNode, expandedScopeIds)) {
    SK_LOGI("Deadlock detected in kernel node, nodeId=%lu", preNode->GetNodeId());
    return true;
  }
  return false;
}

bool LockDetector::CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode, std::unordered_set<uint16_t> &expandedScopeIds) {
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
  if (HasDeadlock(preNode, expandedScopeIds)) {
    SK_LOGI("Deadlock detected in wait node pre-path, waitNodeId=%lu", preNode->GetNodeId());
    return true;
  }
  // Case 4: check node before notify node (notify node not in sk stream) in different stream
  if (notifyNode->GetStreamIdxInGraph() != preNode->GetStreamIdxInGraph() && !IsInSKStream(*notifyNode)) {
    if (HasDeadlock(notifyNode, expandedScopeIds)) {
      SK_LOGI("Deadlock detected in wait node cross-stream path, waitNodeId=%lu, notifyNodeId=%lu",
              preNode->GetNodeId(), notifyNode->GetNodeId());
      return true;
    }
  }
  return false;
}

bool LockDetector::CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode,
                                           std::unordered_set<uint16_t> &expandedScopeIds) {
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
      if (HasDeadlock(preNode, expandedScopeIds)) {
        SK_LOGI("Deadlock detected in notify node path, notifyNodeId=%lu", preNode->GetNodeId());
        return true;
      }
    }
  }
  return false;
}

// LockDetector: maintain core demand and visited nodes while checking the current scope.
void LockDetector::RecordResourceConflict(uint32_t externalCubeNum, uint32_t externalVecNum, uint64_t cubeSourceNodeId,
                                          uint64_t vectorSourceNodeId) {
  resourceConflictInfo_ = {};
  const auto addSource = [this](uint64_t nodeId) {
    const auto *node = graph_->GetNodeById(nodeId);
    if (node == nullptr) {
      return;
    }
    const auto scopeIt = scopeInfos_.find(node->GetScopeId());
    if (scopeIt != scopeInfos_.end() && scopeIt->first != currentScopeId_) {
      resourceConflictInfo_.parallelScopeIds.push_back(scopeIt->first);
    } else {
      resourceConflictInfo_.parallelNodeIds.push_back(nodeId);
    }
  };
  if (deviceRealCubeNum < 0 || static_cast<uint64_t>(superKernelCubeNum) + externalCubeNum > deviceRealCubeNum) {
    addSource(cubeSourceNodeId);
  }
  if (deviceRealVecNum < 0 || static_cast<uint64_t>(superKernelVecNum) + externalVecNum > deviceRealVecNum) {
    addSource(vectorSourceNodeId);
  }
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
      return true;
    }
    SK_LOGD("[lock detector] Candidate SK for node %s conflicts with dependent ops (cube %u>%u or vec %u>%u)",
            curNode->Format().c_str(), superKernelCubeNum, availableCores.first, superKernelVecNum,
            availableCores.second);
    RecordResourceConflict(depOpCubeNum, depOpVecNum, depOpCubeSourceNodeId_, depOpVectorSourceNodeId_);
    return false;
  } else {
    const uint32_t candidateDepCubeNum = std::max(depOpCubeNum, curNodeCubeNum);
    const uint32_t candidateDepVecNum = std::max(depOpVecNum, curNodeVecNum);
    const uint64_t cubeSourceNodeId = curNodeCubeNum > depOpCubeNum ? curNode->GetNodeId() : depOpCubeSourceNodeId_;
    const uint64_t vectorSourceNodeId = curNodeVecNum > depOpVecNum ? curNode->GetNodeId() : depOpVectorSourceNodeId_;
    std::pair<uint32_t, uint32_t> availableCores = GetAvailableCores(false);
    if (HasEnoughDeviceCores(superKernelCubeNum, superKernelVecNum, candidateDepCubeNum, candidateDepVecNum)) {
      depOpCubeNum = candidateDepCubeNum;
      depOpVecNum = candidateDepVecNum;
      depOpCubeSourceNodeId_ = cubeSourceNodeId;
      depOpVectorSourceNodeId_ = vectorSourceNodeId;
      SK_LOGD("[lock detector] Node %s: allocated from device (cube %u, vec %u), new depOp limits: cube %u, vec %u",
              curNode->Format().c_str(), curNodeCubeNum, curNodeVecNum, depOpCubeNum, depOpVecNum);
      return true;
    }
    SK_LOGD(
        "[lock detector] Node %s: insufficient cores for depOp (required: cube %u, vec %u, available: cube %u, "
        "vec %u)",
        curNode->Format().c_str(), curNodeCubeNum, curNodeVecNum, availableCores.first, availableCores.second);
    RecordResourceConflict(candidateDepCubeNum, candidateDepVecNum, cubeSourceNodeId, vectorSourceNodeId);
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
  depOpCubeSourceNodeId_ = INVALID_TASK_ID;
  depOpVectorSourceNodeId_ = INVALID_TASK_ID;
  resourceConflictInfo_ = {};
  superKernelCubeNum = 0;
  superKernelVecNum = 0;
  currentScopeId_ = INVALID_SCOPE_ID;
  scopeCoreInfoChanged_ = false;
  nodeNum = 0;
  kernelNodeNum = 0;
  deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;

  RollbackVisitedState(nodes);
  RollbackVisitedState(tempVisitedNodes);
  candidateScopeNodes_.clear();
  activeConcurrentRanges_.clear();
  skStreamIds.clear();
  skRangeInStream.clear();
  SK_LOGD(
      "[lock detector] Reset: Completed, cleared %zu nodes, %zu tempVisitedNodes, %zu stream IDs, %zu stream ranges",
      nodesCount, tempVisitedNodesCount, streamIdsCount, streamRangesCount);
}

// LockDetector: track the fused node span on each stream.
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

// LockDetector: fuse each node after dispatching by node type and checking Wait dependencies.
bool LockDetector::IsFusible(SuperKernelBaseNode &curNode) {
  // If node already visited (already checked or fused), return true directly without modifying state
  if (curNode.IsVisited()) {
    SK_LOGD("[lock detector] Node %s: already visited, can fuse", curNode.Format().c_str());
    return true;
  }

  SK_LOGI("[lock detector] IsFusible: Checking node %s, current state: nodeNum=%u, kernelNodeNum=%u",
          curNode.Format().c_str(), nodeNum, kernelNodeNum);
  deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;  // Reset before checking
  resourceConflictInfo_ = {};
  const bool canFuse = GetFusibleStatus(curNode);
  // Only modify state if node can be fused
  if (canFuse) {
    skStreamIds.insert(curNode.GetStreamIdxInGraph());
    UpdateSKRangeInStream(curNode);
    nodeNum++;
    curNode.SetVisited(true);
    nodes.emplace_back(curNode.GetNodeId());
    candidateScopeNodes_.push_back(&curNode);

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
  if (currentScopeId_ != INVALID_SCOPE_ID) {
    auto candidate = candidateScopeNodes_;
    candidate.push_back(&curNode);
    activeConcurrentRanges_ = CalculateScopeConcurrentRanges(candidate, currentScopeId_);
    if (concurrentRanges_.IsBeforeConcurrentRange(*notifyNode, activeConcurrentRanges_)) {
      return true;
    }
  }
  // Case 3: notify node is in the same SK stream
  if (IsInSKStream(*notifyNode)) {
    if (!CheckNotifyInSKStream(curNode, *notifyNode)) {
      return false;
    }
  } else if (!concurrentRanges_.HasIntersection(skStreamIds, notifyNode->GetScopeStreamIds())) {
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
    std::unordered_set<uint16_t> expandedScopeIds;
    bool hasDeadlock = HasDeadlock(notifyNode, expandedScopeIds);
    SK_LOGD("[lock detector] Wait node %s: notify %s HasDeadlock=%d", curNode.Format().c_str(),
            notifyNode->Format().c_str(), hasDeadlock);
    if (hasDeadlock) {
      return false;
    }
  }

  if (currentScopeId_ != INVALID_SCOPE_ID && !candidateScopeNodes_.empty()) {
    auto candidate = candidateScopeNodes_;
    candidate.push_back(&curNode);
    const CoreDemand rootDemand{superKernelCubeNum, superKernelVecNum};
    if (!CheckWaitResourceBranches(curNode, currentScopeId_, rootDemand, candidate, {currentScopeId_}, nullptr,
                                   &resourceConflictInfo_)) {
      deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
      return false;
    }
  }

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

// LockDetector: set and restore Notify expansion metadata around scope checks.
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

// ScopeDeadlockDetector: enumerate concurrent scope combinations and check their Wait resource branches.
const SuperKernelScopeInfo *ScopeDeadlockDetector::CheckParallelScopeResources(
    const std::vector<SuperKernelScopeInfo> &scopes, uint64_t &riskWaitId, ResourceConflictInfo *conflictInfo) {
  riskWaitId = INVALID_TASK_ID;
  if (conflictInfo != nullptr) {
    *conflictInfo = {};
  }
  const SuperKernelScopeInfo *riskScope = nullptr;
  scopeInfos_.clear();
  for (const auto &scope : scopes) {
    if (scope.GetScopeCoreInfo().IsValid()) {
      scopeInfos_.emplace(scope.GetScopeId(), ScopeRuntimeInfo::FromScope(scope));
    }
  }
  const auto candidates = CollectWaitScopeCandidates(scopes);
  const auto ranges = BuildScopeConcurrentRanges(scopes);
  std::vector<const SuperKernelScopeInfo *> selected;
  WaitResourceDependencyAnalyzer::CoreDemand selectedDemand;
  std::function<bool(size_t)> check = [&](size_t index) {
    if (index == candidates.size()) {
      return CheckSelectedScopeResourceBranches(selected, ranges, riskWaitId, riskScope, conflictInfo);
    }
    if (!check(index + 1)) {
      return false;
    }
    const auto *candidate = candidates[index];
    for (const auto *scope : selected) {
      if (!concurrentRanges_.CanRunTogether(scope->GetScopeId(), candidate->GetScopeId(), ranges)) {
        return true;
      }
    }
    const uint64_t candidateCube = selectedDemand.cubeCoreNum + candidate->GetScopeCoreInfo().GetCubeNum();
    const uint64_t candidateVec = selectedDemand.vectorCoreNum + candidate->GetScopeCoreInfo().GetVectorNum();
    if (!waitResources_.HasEnoughDeviceCores(candidateCube, candidateVec, 0, 0)) {
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

std::vector<const SuperKernelScopeInfo *> ScopeDeadlockDetector::CollectWaitScopeCandidates(
    const std::vector<SuperKernelScopeInfo> &scopes) const {
  std::vector<const SuperKernelScopeInfo *> candidates;
  for (const auto &scope : scopes) {
    std::unordered_set<uint64_t> memberIds;
    for (const auto *node : scope.GetNodes()) {
      memberIds.insert(node->GetNodeId());
    }
    const bool hasCheckableWait =
        std::any_of(scope.GetNodes().begin(), scope.GetNodes().end(), [this, &memberIds](const auto *node) {
          return node->GetNodeType() == SkNodeType::NODE_WAIT && !waitResources_.ShouldBypassValueWaitDeadlock(*node) &&
                 !IsWaitNotifyInScope(*node, memberIds);
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
                     concurrentRanges_.CalculateScopeConcurrentRanges(scope.GetNodes(), scope.GetScopeId()));
    }
  }
  return ranges;
}

bool ScopeDeadlockDetector::CheckSelectedScopeResourceBranches(
    const std::vector<const SuperKernelScopeInfo *> &selected, const ScopeConcurrentRanges &ranges,
    uint64_t &riskWaitId, const SuperKernelScopeInfo *&riskScope, ResourceConflictInfo *conflictInfo) {
  if (selected.size() < 2) {
    return true;
  }
  std::unordered_set<uint16_t> residents;
  for (const auto *scope : selected) {
    residents.insert(scope->GetScopeId());
  }
  for (const auto *scope : selected) {
    const auto scopeId = scope->GetScopeId();
    const WaitResourceDependencyAnalyzer::CoreDemand rootDemand{scope->GetScopeCoreInfo().GetCubeNum(),
                                                                scope->GetScopeCoreInfo().GetVectorNum()};
    for (const auto *node : scope->GetNodes()) {
      if (node->GetNodeType() == SkNodeType::NODE_WAIT && !waitResources_.ShouldBypassValueWaitDeadlock(*node) &&
          !waitResources_.CheckWaitResourceBranches(*node, scopeId, rootDemand, scope->GetNodes(), residents, &ranges,
                                                    conflictInfo)) {
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
