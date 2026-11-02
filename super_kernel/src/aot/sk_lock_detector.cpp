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
#include <map>

int64_t LockDetector::deviceRealCubeNum = 0;
int64_t LockDetector::deviceRealVecNum = 0;

void LockDetector::Init(SuperKernelGraph &graph) {
  SK_LOGD("[lock detector] LockDetector::Init: Initializing lock detector");
  nodes.clear();
  candidateScopeNodes_.clear();
  activeConcurrentRanges_.clear();
  depOpCubeNum = 0;
  depOpVecNum = 0;
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
  scopeInfo.memberNodes = scope.GetNodes();
  scopeInfo.hasWait = std::any_of(scopeInfo.memberNodes.begin(), scopeInfo.memberNodes.end(),
                                  [](const auto *node) { return node->GetNodeType() == SkNodeType::NODE_WAIT; });
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

bool LockDetector::IsInSKStream(const SuperKernelBaseNode &node) const {
  return std::find(skStreamIds.begin(), skStreamIds.end(), node.GetStreamIdxInGraph()) != skStreamIds.end();
}

SuperKernelBaseNode *LockDetector::GetScopeBoundary(const ConcurrentRangeState &rangeContext, uint32_t scopeId,
                                                    uint32_t streamId, bool first) const {
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

uint32_t LockDetector::GetTrackedScopeId(const ConcurrentRangeState &rangeContext,
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

void LockDetector::InitializeConcurrentRangeState(const std::vector<SuperKernelBaseNode *> &candidate,
                                                  uint16_t candidateScopeId, ConcurrentRangeState &rangeContext) const {
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
  rangeContext.ranges = rangeContext.candidateBoundaries;
  const auto &headNodes = graph_->GetHeadNodes();
  const auto &tailNodes = graph_->GetTailNodes();
  for (uint32_t streamId = 0; streamId < headNodes.size(); ++streamId) {
    if (headNodes[streamId] != INVALID_TASK_ID && rangeContext.ranges.count(streamId) == 0) {
      rangeContext.ranges.emplace(streamId, ConcurrentNodeRange{headNodes[streamId], tailNodes[streamId]});
    }
  }
}

void LockDetector::TightenConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t streamId,
                                             const SuperKernelBaseNode *boundary, bool left) const {
  if (GetScopeBoundary(rangeContext, CANDIDATE_CONCURRENT_SCOPE, streamId, true) != nullptr) {
    return;
  }
  const auto rangeIt = rangeContext.ranges.find(streamId);
  if (rangeIt == rangeContext.ranges.end() || rangeIt->second.firstNodeId == INVALID_TASK_ID) {
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

void LockDetector::TightenScopeConcurrentBoundary(ConcurrentRangeState &rangeContext, uint32_t scopeId,
                                                  uint32_t entryStreamId, bool left, bool eventBoundary) const {
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

void LockDetector::UpdateConcurrentBoundary(ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex,
                                            bool left) const {
  if (vertex.kind == ConcurrentVertex::Kind::SCOPE) {
    return;
  }
  const auto *node = graph_->GetNodeById(vertex.nodeId);
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
  if (scopeIt != scopeInfos_.end() && scopeIt->second.streamIds.size() > 1 && vertex.scopeId != scopeId) {
    TightenScopeConcurrentBoundary(rangeContext, scopeId, node->GetStreamIdxInGraph(), left, false);
  }
}

void LockDetector::QueueConcurrentBoundarySeeds(const ConcurrentRangeState &rangeContext, bool left,
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

void LockDetector::AppendScopeVertices(const ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex,
                                       bool left, std::deque<ConcurrentVertex> &pending) const {
  const auto scopeIt = scopeInfos_.find(vertex.scopeId);
  if (scopeIt == scopeInfos_.end()) {
    return;
  }
  for (const auto streamId : scopeIt->second.streamIds) {
    const auto *boundary = GetScopeBoundary(rangeContext, vertex.scopeId, streamId, !left);
    if (boundary != nullptr) {
      pending.push_back({ConcurrentVertex::Kind::NODE, vertex.scopeId, boundary->GetNodeId()});
    }
  }
}

void LockDetector::AppendConcurrentStreamVertex(const ConcurrentRangeState &rangeContext,
                                                const SuperKernelBaseNode &node, bool left,
                                                const ConcurrentVertex &vertex,
                                                std::deque<ConcurrentVertex> &pending) const {
  const auto scopeId = GetTrackedScopeId(rangeContext, node);
  const auto scopeIt = scopeInfos_.find(scopeId);
  if (scopeIt != scopeInfos_.end() && scopeIt->second.streamIds.size() > 1 && vertex.scopeId != scopeId) {
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

void LockDetector::AppendForwardVertices(const ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex,
                                         std::deque<ConcurrentVertex> &pending) const {
  if (vertex.kind == ConcurrentVertex::Kind::SCOPE) {
    AppendScopeVertices(rangeContext, vertex, false, pending);
    return;
  }
  const auto *node = graph_->GetNodeById(vertex.nodeId);
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

void LockDetector::AppendReverseVertices(const ConcurrentRangeState &rangeContext, const ConcurrentVertex &vertex,
                                         std::deque<ConcurrentVertex> &pending) const {
  if (vertex.kind == ConcurrentVertex::Kind::SCOPE) {
    AppendScopeVertices(rangeContext, vertex, true, pending);
    return;
  }
  const auto *node = graph_->GetNodeById(vertex.nodeId);
  if (node == nullptr) {
    return;
  }
  if (node->GetNodeType() == SkNodeType::NODE_WAIT && node->GetCorrespondingNotifyNodeId() != INVALID_TASK_ID) {
    pending.push_back({ConcurrentVertex::Kind::EVENT_NODE, INVALID_SCOPE_ID, node->GetCorrespondingNotifyNodeId()});
  }
  AppendConcurrentStreamVertex(rangeContext, *node, true, vertex, pending);
}

void LockDetector::TraverseConcurrentBoundary(ConcurrentRangeState &rangeContext, bool left) const {
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

void LockDetector::CalculateLeftConcurrentBoundary(ConcurrentRangeState &rangeContext) const {
  TraverseConcurrentBoundary(rangeContext, true);
}

void LockDetector::CalculateRightConcurrentBoundary(ConcurrentRangeState &rangeContext) const {
  TraverseConcurrentBoundary(rangeContext, false);
}

std::unordered_map<uint32_t, LockDetector::ConcurrentNodeRange> LockDetector::CalculateScopeConcurrentRanges(
    const std::vector<SuperKernelBaseNode *> &candidate, uint16_t candidateScopeId) const {
  ConcurrentRangeState rangeContext;
  InitializeConcurrentRangeState(candidate, candidateScopeId, rangeContext);
  CalculateLeftConcurrentBoundary(rangeContext);
  CalculateRightConcurrentBoundary(rangeContext);
  return rangeContext.ranges;
}

bool LockDetector::IsBeforeConcurrentRange(const SuperKernelBaseNode &node,
                                           const std::unordered_map<uint32_t, ConcurrentNodeRange> &range) const {
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

bool LockDetector::CanRunTogether(uint16_t firstScopeId, uint16_t secondScopeId,
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

void LockDetector::QueueScopeWaitNotifies(uint16_t scopeId, ResourceSearchState &state) const {
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
        state.pending.push_back(notifyId);
      }
    }
  }
}

bool LockDetector::ResourceSearchHasRisk(const ResourceSearchState &state) const {
  uint64_t externalCube = 0;
  uint64_t externalVec = 0;
  for (const auto &[streamId, demand] : state.externalByStream) {
    externalCube = std::max(externalCube, demand.cube);
    externalVec = std::max(externalVec, demand.vec);
  }
  for (const auto &[scopeId, demand] : state.externalScopes) {
    externalCube = std::max(externalCube, demand.cube);
    externalVec = std::max(externalVec, demand.vec);
  }
  const bool hasRisk =
      !HasEnoughDeviceCores(state.residentDemand.cube, state.residentDemand.vec, externalCube, externalVec);
  if (hasRisk) {
    SK_LOGI(
        "[ParallelResource] Resource risk: residentCube=%lu, externalCube=%lu, totalCube=%lu, deviceCube=%ld, "
        "residentVec=%lu, externalVec=%lu, totalVec=%lu, deviceVec=%ld",
        state.residentDemand.cube, externalCube, state.residentDemand.cube + externalCube, deviceRealCubeNum,
        state.residentDemand.vec, externalVec, state.residentDemand.vec + externalVec, deviceRealVecNum);
    for (const auto &[streamId, demand] : state.externalByStream) {
      SK_LOGI("[ParallelResource] External stream %u: cube=%lu, vec=%lu", streamId, demand.cube, demand.vec);
    }
    for (const auto &[scopeId, demand] : state.externalScopes) {
      SK_LOGI("[ParallelResource] External scope %u: cube=%lu, vec=%lu", scopeId, demand.cube, demand.vec);
    }
  }
  return hasRisk;
}

void LockDetector::PrepareResourceSearch(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                         const ResourceDemand &rootDemand,
                                         const std::vector<SuperKernelBaseNode *> &rootNodes,
                                         const std::unordered_set<uint16_t> &initialResidents,
                                         ScopeConcurrentRanges &ranges, ResourceSearchState &state,
                                         const ScopeConcurrentRanges *precomputedRanges) {
  if (precomputedRanges == nullptr) {
    ranges.emplace(rootScopeId, CalculateScopeConcurrentRanges(rootNodes, rootScopeId));
    for (const auto &[scopeId, info] : scopeInfos_) {
      if (scopeId != rootScopeId && info.hasWait) {
        ranges.emplace(scopeId, CalculateScopeConcurrentRanges(info.memberNodes, scopeId));
      }
    }
  }
  state.pending.push_back(waitNode.GetCorrespondingNotifyNodeId());
  state.residentScopes = initialResidents;
  state.residentScopes.insert(rootScopeId);
  state.residentDemand = rootDemand;
  for (const auto scopeId : state.residentScopes) {
    const auto scopeIt = scopeInfos_.find(scopeId);
    if (scopeId != rootScopeId && scopeIt != scopeInfos_.end()) {
      state.residentDemand.cube += scopeIt->second.coreInfo.GetCubeNum();
      state.residentDemand.vec += scopeIt->second.coreInfo.GetVectorNum();
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

bool LockDetector::CanStartWaitScope(uint16_t scopeId, const ScopeRuntimeInfo &scope, uint16_t rootScopeId,
                                     const std::vector<SuperKernelBaseNode *> &rootNodes,
                                     const ScopeConcurrentRanges &ranges, const ResourceSearchState &state) const {
  for (const auto residentId : state.residentScopes) {
    if (residentId != rootScopeId) {
      if (!CanRunTogether(residentId, scopeId, ranges)) {
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

void LockDetector::QueueStartedScopeBranch(uint16_t scopeId, const ScopeRuntimeInfo &scope,
                                           const ResourceSearchState &state,
                                           std::vector<ResourceSearchState> &branches) const {
  ResourceSearchState started = state;
  started.residentScopes.insert(scopeId);
  started.residentDemand.cube += scope.coreInfo.GetCubeNum();
  started.residentDemand.vec += scope.coreInfo.GetVectorNum();
  started.externalScopes.erase(scopeId);
  for (const auto scopeStream : scope.streamIds) {
    started.occupiedStreams[scopeStream] = scopeId;
    started.externalByStream.erase(scopeStream);
  }
  for (auto it = started.externalScopes.begin(); it != started.externalScopes.end();) {
    const auto otherIt = scopeInfos_.find(it->first);
    if (otherIt != scopeInfos_.end() && HasIntersection(scope.streamIds, otherIt->second.streamIds)) {
      it = started.externalScopes.erase(it);
    } else {
      ++it;
    }
  }
  QueueScopeWaitNotifies(scopeId, started);
  branches.push_back(std::move(started));
}

void LockDetector::ExpandUnstartedScope(uint16_t scopeId, const ScopeRuntimeInfo &scope,
                                        ResourceSearchState &state) const {
  state.externalScopes[scopeId] = {scope.coreInfo.GetCubeNum(), scope.coreInfo.GetVectorNum()};
  for (const auto &[headStream, headId] : scope.headNodeIds) {
    const auto *head = graph_->GetNodeById(headId);
    if (head != nullptr && head->GetPreNodeId() != INVALID_TASK_ID) {
      state.pending.push_back(head->GetPreNodeId());
    }
  }
  if (scope.hasWait) {
    QueueScopeWaitNotifies(scopeId, state);
  }
}

bool LockDetector::ProcessResourceSearchNode(uint64_t nodeId, uint16_t rootScopeId,
                                             const std::vector<SuperKernelBaseNode *> &rootNodes,
                                             const ScopeConcurrentRanges &ranges, ResourceSearchState &state,
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
  if (!state.visited.insert(nodeId).second || IsBeforeConcurrentRange(*node, ranges.at(rootScopeId))) {
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
      const uint64_t startedCube = state.residentDemand.cube + scope.coreInfo.GetCubeNum();
      const uint64_t startedVec = state.residentDemand.vec + scope.coreInfo.GetVectorNum();
      if (HasEnoughDeviceCores(startedCube, startedVec, 0, 0)) {
        QueueStartedScopeBranch(scopeId, scope, state, branches);
      }
    }
    ExpandUnstartedScope(scopeId, scope, state);
    return true;
  }
  auto &demand = state.externalByStream[streamId];
  demand.cube = std::max<uint64_t>(demand.cube, node->GetCubeNum());
  demand.vec = std::max<uint64_t>(demand.vec, node->GetVecNum());
  if (node->GetPreNodeId() != INVALID_TASK_ID) {
    state.pending.push_back(node->GetPreNodeId());
  }
  if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
    state.pending.push_back(node->GetCorrespondingNotifyNodeId());
  }
  return true;
}

bool LockDetector::CheckWaitResourceBranches(const SuperKernelBaseNode &waitNode, uint16_t rootScopeId,
                                             const ResourceDemand &rootDemand,
                                             const std::vector<SuperKernelBaseNode *> &rootNodes,
                                             const std::unordered_set<uint16_t> &initialResidents,
                                             const ScopeConcurrentRanges *precomputedRanges) {
  ScopeConcurrentRanges calculatedRanges;
  ResourceSearchState initial;
  PrepareResourceSearch(waitNode, rootScopeId, rootDemand, rootNodes, initialResidents, calculatedRanges, initial,
                        precomputedRanges);
  const auto &ranges = precomputedRanges == nullptr ? calculatedRanges : *precomputedRanges;
  if (!HasEnoughDeviceCores(initial.residentDemand.cube, initial.residentDemand.vec, 0, 0)) {
    SK_LOGI("[ParallelResource] Wait %lu: resident scopes require cube=%lu, vec=%lu; device has cube=%ld, vec=%ld",
            waitNode.GetNodeId(), initial.residentDemand.cube, initial.residentDemand.vec, deviceRealCubeNum,
            deviceRealVecNum);
    return false;
  }

  std::vector<ResourceSearchState> branches;
  branches.push_back(std::move(initial));
  while (!branches.empty()) {
    ResourceSearchState state = std::move(branches.back());
    branches.pop_back();
    while (!state.pending.empty()) {
      const auto nodeId = state.pending.back();
      state.pending.pop_back();
      if (!ProcessResourceSearchNode(nodeId, rootScopeId, rootNodes, ranges, state, branches)) {
        SK_LOGE("[ParallelResource] Wait %lu: failed to search dependency node %lu", waitNode.GetNodeId(), nodeId);
        return false;
      }
    }
    if (ResourceSearchHasRisk(state)) {
      SK_LOGI("[ParallelResource] Wait %lu in scope %u has resource risk", waitNode.GetNodeId(), rootScopeId);
      return false;
    }
  }
  return true;
}

bool LockDetector::HasDeadlock(SuperKernelBaseNode *curNode, DeadlockSearchState &state) {
  curNode->SetVisited(true);
  tempVisitedNodes.emplace_back(curNode->GetNodeId());
  const auto scopeIt = scopeInfos_.find(curNode->GetScopeId());
  if (curNode->GetScopeId() != currentScopeId_ && scopeIt != scopeInfos_.end()) {
    if (!state.expandedScopeIds.insert(curNode->GetScopeId()).second) {
      return false;
    }
    return CheckScopeDeadlock(scopeIt->second, state);
  }
  return HasDeadlockPredecessor(curNode, state);
}

bool LockDetector::CheckScopeDeadlock(const ScopeRuntimeInfo &scopeInfo, DeadlockSearchState &state) {
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
        HasDeadlock(notify, state)) {
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
    if (HasDeadlockPredecessor(head, state)) {
      return true;
    }
  }
  return false;
}

bool LockDetector::HasDeadlockPredecessor(SuperKernelBaseNode *curNode, DeadlockSearchState &state) {
  if (curNode->GetPreNodeId() == INVALID_TASK_ID) {
    return false;
  }

  const uint64_t preNodeId = curNode->GetPreNodeId();
  SuperKernelBaseNode *preNode = graph_->GetNodeById(preNodeId);
  if (preNode == nullptr) {
    SK_LOGE("[lock detector] HasDeadlock: preNode %lu not found for curNode %lu", preNodeId, curNode->GetNodeId());
    return false;
  }
  if (IsBeforeConcurrentRange(*preNode, activeConcurrentRanges_)) {
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

  const auto scopeIt = scopeInfos_.find(preNode->GetScopeId());
  if (preNode->GetScopeId() != currentScopeId_ && scopeIt != scopeInfos_.end()) {
    return HasDeadlock(preNode, state);
  }

  bool hasDeadlock = true;
  switch (preNode->GetNodeType()) {
    case SkNodeType::NODE_KERNEL:
      hasDeadlock = preNode->IsScopeNode() ? HasDeadlock(preNode, state) : CheckKernelNodeDeadlock(preNode, state);
      break;
    case SkNodeType::NODE_WAIT:
      hasDeadlock = CheckWaitNodeDeadlock(preNode, state);
      break;
    case SkNodeType::NODE_NOTIFY:
      hasDeadlock = CheckNotifyNodeDeadlock(preNode, state);
      break;
    case SkNodeType::NODE_DEFAULT:
      hasDeadlock = HasDeadlock(preNode, state);
      break;
    case SkNodeType::NODE_RESET:
      hasDeadlock = HasDeadlock(preNode, state);
      break;
    default:
      SK_LOGD("nodeId: %u, unsupported node type %u in HasDeadlock", preNode->GetNodeId(), preNode->GetNodeType());
      deadlockReason_ = DeadlockFailReason::NO_SUPPORT_NODE;
      break;
  }

  return hasDeadlock;
}

bool LockDetector::CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode, DeadlockSearchState &state) {
  if (!HasEnoughCores(preNode, false)) {
    SK_LOGI("Not enough cores for kernel node, nodeId=%lu, requiredCube=%u, requiredVec=%u", preNode->GetNodeId(),
            preNode->GetCubeNum(), preNode->GetVecNum());
    deadlockReason_ = DeadlockFailReason::KERNEL_INSUFFICIENT_CORES;
    return true;
  }
  if (HasDeadlock(preNode, state)) {
    SK_LOGI("Deadlock detected in kernel node, nodeId=%lu", preNode->GetNodeId());
    return true;
  }
  return false;
}

bool LockDetector::CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode, DeadlockSearchState &state) {
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
  if (HasDeadlock(preNode, state)) {
    SK_LOGI("Deadlock detected in wait node pre-path, waitNodeId=%lu", preNode->GetNodeId());
    return true;
  }
  // Case 4: check node before notify node (notify node not in sk stream) in different stream
  if (notifyNode->GetStreamIdxInGraph() != preNode->GetStreamIdxInGraph() && !IsInSKStream(*notifyNode)) {
    if (HasDeadlock(notifyNode, state)) {
      SK_LOGI("Deadlock detected in wait node cross-stream path, waitNodeId=%lu, notifyNodeId=%lu",
              preNode->GetNodeId(), notifyNode->GetNodeId());
      return true;
    }
  }
  return false;
}

bool LockDetector::CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode, DeadlockSearchState &state) {
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
      if (HasDeadlock(preNode, state)) {
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
      return true;
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
  if (currentScopeId_ != INVALID_SCOPE_ID) {
    auto candidate = candidateScopeNodes_;
    candidate.push_back(&curNode);
    activeConcurrentRanges_ = CalculateScopeConcurrentRanges(candidate, currentScopeId_);
    if (IsBeforeConcurrentRange(*notifyNode, activeConcurrentRanges_)) {
      return true;
    }
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
    DeadlockSearchState searchState;
    bool hasDeadlock = HasDeadlock(notifyNode, searchState);
    SK_LOGD("[lock detector] Wait node %s: notify %s HasDeadlock=%d", curNode.Format().c_str(),
            notifyNode->Format().c_str(), hasDeadlock);
    if (hasDeadlock) {
      return false;
    }
  }

  if (currentScopeId_ != INVALID_SCOPE_ID && !candidateScopeNodes_.empty()) {
    auto candidate = candidateScopeNodes_;
    candidate.push_back(&curNode);
    const ResourceDemand rootDemand{superKernelCubeNum, superKernelVecNum};
    if (!CheckWaitResourceBranches(curNode, currentScopeId_, rootDemand, candidate, {currentScopeId_})) {
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
