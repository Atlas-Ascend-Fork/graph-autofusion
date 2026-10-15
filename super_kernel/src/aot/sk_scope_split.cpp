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
 * \file sk_scope_split.cpp
 * \brief Implementation of multi-pass scope splitting
 */

#include "sk_scope_split.h"
#include "sk_common.h"
#include <algorithm>
#include <unordered_map>

namespace {
bool IsExecutableKernelNode(const SuperKernelBaseNode *node) {
  return node != nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL && !node->IsScopeNode();
}

ScopeCoreInfo BuildDebugPerOpScopeCoreInfo(const SuperKernelBaseNode &node, int64_t maxDeviceCubeNum,
                                           int64_t maxDeviceVectorNum) {
  if (node.RequiresExactCoreMatch()) {
    if (node.GetCubeNum() == 0 && node.GetVecNum() != 0 && (node.GetVecNum() & 1U) != 0U) {
      return {SkKernelType::AIV_ONLY, node.GetVecNum()};
    } else {
      const uint32_t numBlocks = node.GetCubeNum() == 0 ? node.GetVecNum() / 2U : node.GetCubeNum();
      return {SkKernelType::MIX_AIC_1_2, numBlocks};
    }
  } else {
    const int64_t numBlocks = std::min(maxDeviceCubeNum, maxDeviceVectorNum / 2);
    return {SkKernelType::MIX_AIC_1_2, static_cast<uint32_t>(numBlocks)};
  }
}
}  // namespace

bool ScopeCoreInfoCalculator::UpdateScopeCoreConstraints(const SuperKernelBaseNode &node,
                                                         ScopeCoreConstraints &constraints) {
  constraints.minCubeNum = std::max(constraints.minCubeNum, node.GetCubeNum());
  constraints.minVectorNum = std::max(constraints.minVectorNum, node.GetVecNum());
  constraints.requireMix12 = constraints.requireMix12 || node.GetKernelType() == SkKernelType::MIX_AIC_1_2;

  if (!node.RequiresExactCoreMatch()) {
    return true;
  }

  const bool vectorOnly = node.GetCubeNum() == 0 && node.GetVecNum() != 0;
  auto &exactCoreNum = vectorOnly ? constraints.exactVectorNum : constraints.exactCubeNum;
  const uint32_t taskCoreNum = vectorOnly ? node.GetVecNum() : node.GetCubeNum();
  if (exactCoreNum.has_value() && exactCoreNum.value() != taskCoreNum) {
    return false;
  }
  exactCoreNum = taskCoreNum;
  return true;
}

bool ScopeCoreInfoCalculator::GetScopeCoreCandidate(const ScopeCoreConstraints &constraints,
                                                    ScopeCoreInfo &scopeCoreInfo) const {
  const auto getCandidates = [](uint32_t minCubeNum, uint32_t minVectorNum, bool requireMix12) {
    if (minCubeNum == 0 && minVectorNum == 0) {
      return std::vector<ScopeCoreInfo>{{}};
    } else if (requireMix12) {
      const uint32_t vectorBlocks = static_cast<uint32_t>((static_cast<uint64_t>(minVectorNum) + 1U) >> 1U);
      return std::vector<ScopeCoreInfo>{{SkKernelType::MIX_AIC_1_2, std::max(minCubeNum, vectorBlocks)}};
    } else if (minCubeNum == 0 && minVectorNum != 0) {
      return std::vector<ScopeCoreInfo>{{SkKernelType::AIV_ONLY, minVectorNum}};
    } else if (minCubeNum != 0 && minVectorNum == 0) {
      return std::vector<ScopeCoreInfo>{{SkKernelType::AIC_ONLY, minCubeNum}};
    } else {
      const uint32_t vectorBlocks = static_cast<uint32_t>((static_cast<uint64_t>(minVectorNum) + 1U) >> 1U);
      const ScopeCoreInfo mix12{SkKernelType::MIX_AIC_1_2, std::max(minCubeNum, vectorBlocks)};
      const ScopeCoreInfo mix11{SkKernelType::MIX_AIC_1_1, std::max(minCubeNum, minVectorNum)};
      return minVectorNum <= minCubeNum ? std::vector<ScopeCoreInfo>{mix11, mix12}
                                        : std::vector<ScopeCoreInfo>{mix12, mix11};
    }
  };

  const auto isValidCandidate = [this, &constraints](const ScopeCoreInfo &candidate) {
    if (!candidate.IsValid()) {
      return false;
    }
    const uint32_t cubeNum = candidate.GetCubeNum();
    const uint32_t vectorNum = candidate.GetVectorNum();
    if (cubeNum > maxDeviceCubeNum_ || vectorNum > maxDeviceVectorNum_ || constraints.minCubeNum > cubeNum ||
        constraints.minVectorNum > vectorNum) {
      return false;
    }
    return (!constraints.exactCubeNum.has_value() || constraints.exactCubeNum.value() == cubeNum) &&
           (!constraints.exactVectorNum.has_value() || constraints.exactVectorNum.value() == vectorNum);
  };

  const auto candidates = getCandidates(constraints.minCubeNum, constraints.minVectorNum, constraints.requireMix12);
  for (const auto &candidate : candidates) {
    if (isValidCandidate(candidate)) {
      scopeCoreInfo = candidate;
      return true;
    }
  }

  // getCandidates has already applied the required SK type, so the first entry is the attempted shape for diagnostics.
  scopeCoreInfo = candidates.front();
  return false;
}

bool ScopeCoreInfoCalculator::GetValidScopeCoreInfo(const std::vector<SuperKernelBaseNode *> &nodes) const {
  ScopeCoreInfo scopeCoreInfo;
  return GetValidScopeCoreInfo(nodes, scopeCoreInfo);
}

bool ScopeCoreInfoCalculator::GetValidScopeCoreInfo(const std::vector<SuperKernelBaseNode *> &nodes,
                                                    ScopeCoreInfo &scopeCoreInfo) const {
  ScopeCoreConstraints constraints;
  for (const auto *node : nodes) {
    if (!IsExecutableKernelNode(node)) {
      continue;
    }
    if (!UpdateScopeCoreConstraints(*node, constraints)) {
      GetScopeCoreCandidate(constraints, scopeCoreInfo);
      return false;
    }
  }
  return GetScopeCoreCandidate(constraints, scopeCoreInfo);
}

bool ScopeCoreInfoCalculator::UpdateScopeCoreInfo(const SuperKernelBaseNode &node, ScopeCoreInfo &scopeCoreInfo) {
  ScopeCoreConstraints nextConstraints = constraints_;
  if (!UpdateScopeCoreConstraints(node, nextConstraints)) {
    GetScopeCoreCandidate(nextConstraints, scopeCoreInfo);
    return false;
  }
  if (!GetScopeCoreCandidate(nextConstraints, scopeCoreInfo)) {
    return false;
  }
  constraints_ = std::move(nextConstraints);
  return true;
}

void ScopeCoreInfoCalculator::Reset() {
  constraints_ = {};
}

// ============ EventOnlyStreamRemovePass Implementation ============

EventOnlyStreamRemovePass::EventOnlyStreamRemovePass(SuperKernelGraph &inputGraph)
    : ScopeSplitPass(inputGraph), markedCount_(0) {}

bool EventOnlyStreamRemovePass::IsEventNode(SuperKernelBaseNode *node) const {
  if (node == nullptr) {
    return false;
  }

  SkNodeType nodeType = node->GetNodeType();
  return (nodeType == SkNodeType::NODE_NOTIFY || nodeType == SkNodeType::NODE_WAIT ||
          nodeType == SkNodeType::NODE_RESET || nodeType == SkNodeType::NODE_MEMORY_WRITE ||
          nodeType == SkNodeType::NODE_MEMORY_WAIT);
}

void EventOnlyStreamRemovePass::CollectNodesPerStream(
    const SuperKernelScopeInfo &scope, std::unordered_map<uint32_t, std::vector<SuperKernelBaseNode *>> &streamNodes) {
  streamNodes.clear();
  for (auto *node : scope.GetNodes()) {
    if (node == nullptr) {
      continue;
    }
    uint32_t streamIdx = node->GetStreamIdxInGraph();
    streamNodes[streamIdx].push_back(node);
  }
}

bool EventOnlyStreamRemovePass::IsStreamAllEventNodes(const std::vector<SuperKernelBaseNode *> &nodes) const {
  if (nodes.empty()) {
    return false;
  }
  for (auto *node : nodes) {
    if (node == nullptr) {
      continue;
    }
    if (!IsEventNode(node)) {
      return false;
    }
  }
  return true;
}

uint32_t EventOnlyStreamRemovePass::ProcessScope(SuperKernelScopeInfo &scope) {
  if (scope.GetNodes().empty()) {
    return 0;
  }

  std::unordered_map<uint32_t, std::vector<SuperKernelBaseNode *>> streamNodes;
  CollectNodesPerStream(scope, streamNodes);

  uint32_t scopeMarkedCount = 0;

  // Identify streams that have all event nodes and mark them as non-fusible
  for (const auto &pair : streamNodes) {
    uint32_t streamIdx = pair.first;
    const auto &nodes = pair.second;

    if (IsStreamAllEventNodes(nodes)) {
      // Mark all event nodes in this stream as non-fusible
      for (auto *node : nodes) {
        if (node != nullptr && node->IsFusible()) {
          node->SetIsFusible(false);
          node->SetFusionFailReason(FusionFailReason::ISOLATED_EVENT);
          scopeMarkedCount++;
          SK_LOGI("[EventOnlyStreamRemove] Marked event node %lu in stream %u as non-fusible", node->GetNodeId(),
                  streamIdx);
        }
      }
      SK_LOGI("[EventOnlyStreamRemove] Stream %u in scope has all event nodes (%zu nodes), marked as non-fusible",
              streamIdx, nodes.size());
    }
  }

  return scopeMarkedCount;
}

bool EventOnlyStreamRemovePass::Run(std::vector<SuperKernelScopeInfo> &scopes) {
  SK_LOGI("[EventOnlyStreamRemove] %s pass starting execution", GetName().c_str());

  scopeSplitResultReporter_->CaptureResplitScopes(scopes);
  markedCount_ = 0;
  SK_LOGI("[EventOnlyStreamRemove] processing %zu scopes", scopes.size());

  for (size_t i = 0; i < scopes.size(); ++i) {
    uint32_t scopeMarked = ProcessScope(scopes[i]);
    markedCount_ += scopeMarked;

    SK_LOGI("[EventOnlyStreamRemove] Scope %zu: after processing, %zu nodes, %zu streams, marked %u nodes", i,
            scopes[i].GetNodes().size(), scopes[i].GetScopeStreamInfos().size(), scopeMarked);
  }

  // If any nodes were marked as non-fusible, signal that re-split is needed
  // and clear all scopes so they will be regenerated
  if (markedCount_ > 0) {
    SK_LOGI("[EventOnlyStreamRemove] Marked %u event-only stream nodes as non-fusible, requesting re-split",
            markedCount_);
    scopes.clear();  // Clear scopes so they will be regenerated
    RequestResplit();
  }

  SK_LOGI("[EventOnlyStreamRemove] %s pass completed, marked %u event-only stream nodes total", GetName().c_str(),
          markedCount_);

  return true;
}

// ============ PerOpMaxCoreSplitPass Implementation ============

PerOpMaxCoreSplitPass::PerOpMaxCoreSplitPass(SuperKernelGraph &inputGraph) : ScopeSplitPass(inputGraph) {}

bool PerOpMaxCoreSplitPass::Run(std::vector<SuperKernelScopeInfo> &scopes) {
  SK_LOGI("[PerOpMaxCore] %s pass starting", GetName().c_str());
  scopes.clear();

  int64_t maxDeviceCubeNum = 0;
  int64_t maxDeviceVectorNum = 0;
  if (GetDeviceCoreNums(maxDeviceCubeNum, maxDeviceVectorNum) != ACL_SUCCESS) {
    return false;
  }

  const auto &headNodes = graph_.GetHeadNodes();
  for (uint64_t headNodeId : headNodes) {
    uint64_t nodeId = headNodeId;
    while (nodeId != INVALID_TASK_ID) {
      SuperKernelBaseNode *node = graph_.GetNodeById(nodeId);
      if (node == nullptr) {
        break;
      }

      if (node->GetNodeType() == SkNodeType::NODE_KERNEL && node->IsFusible()) {
        SuperKernelScopeInfo scope;
        scope.AddNode(node);
        scope.SetScopeBitFlags(node->GetScopeBitFlags());
        RebuildStreamInfos(scope);
        scope.SetScopeCoreInfo(BuildDebugPerOpScopeCoreInfo(*node, maxDeviceCubeNum, maxDeviceVectorNum));
        scopes.push_back(std::move(scope));
        SK_LOGI("[PerOpMaxCore] Created scope for kernel %lu (scopeIdx=%zu)", node->GetNodeId(), scopes.size() - 1);
      }
      nodeId = node->GetNextNodeId();
    }
  }

  SK_LOGI("[PerOpMaxCore] Completed, %zu scopes", scopes.size());
  return true;
}

// ============ ScopeSplitPass Base Class Implementation ============

void ScopeSplitPass::PrintScopeDetails(const std::vector<SuperKernelScopeInfo> &scopes, const SuperKernelGraph &graph,
                                       const char *passName) {
  SK_LOGI("========== Scope split results begin: pass=%s, totalScopes=%zu ==========", passName, scopes.size());
  for (size_t i = 0; i < scopes.size(); ++i) {
    const auto &scope = scopes[i];
    std::string scopeNames = GetScopeNamesFromBitFlags(scope.GetScopeBitFlags(), graph);
    SK_LOGI("Scope %zu (scopeId=%u): %zu nodes, %zu streams, scopeBitFlags=%s, scopeNames=[%s]", i, scope.GetScopeId(),
            scope.GetNodes().size(), scope.GetScopeStreamInfos().size(),
            graph.BitsetToString(scope.GetScopeBitFlags()).c_str(), scopeNames.c_str());
    SK_LOGI("  BreakInfo: %s", scope.GetBreakInfo().Format().c_str());
    PrintScopeNodes(i, scope);
    PrintScopeStreamInfos(i, scope);
  }
  SK_LOGI("========== Scope split results end: pass=%s ==========", passName);
}

void ScopeSplitPass::PrintScopeResults(const std::vector<SuperKernelScopeInfo> &scopes, const SuperKernelGraph &graph,
                                       const char *passName) {
  // Log to dedicated file
  SK_LOG_CONTEXT_SIMPLE("sk_scope_split.log");
  PrintScopeDetails(scopes, graph, passName);
}

std::string ScopeSplitPass::GetScopeNamesFromBitFlags(const std::bitset<MAX_SCOPE_NUM> &scopeBitFlags,
                                                      const SuperKernelGraph &graph) {
  std::string scopeNames;
  for (size_t bit = 0; bit < MAX_SCOPE_NUM && bit < scopeBitFlags.size(); ++bit) {
    if (scopeBitFlags.test(bit)) {
      std::string name;
      if (graph.GetScopeNameByIdx(static_cast<uint32_t>(bit), name)) {
        if (!scopeNames.empty()) scopeNames += "_";
        scopeNames += name;
      }
    }
  }
  return scopeNames.empty() ? "none" : scopeNames;
}

void ScopeSplitPass::PrintScopeNodes(size_t scopeIdx, const SuperKernelScopeInfo &scope) {
  const auto &nodes = scope.GetNodes();
  SK_LOGI("  Scope %zu nodes: %zu", scopeIdx, nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    SK_LOGI("    [%zu] %s", i, nodes[i]->Format().c_str());
  }
}

void ScopeSplitPass::PrintScopeStreamInfos(size_t scopeIdx, const SuperKernelScopeInfo &scope) {
  const auto &scopeStreamInfos = scope.GetScopeStreamInfos();
  for (size_t j = 0; j < scopeStreamInfos.size(); ++j) {
    const auto &streamInfo = scopeStreamInfos[j];
    SK_LOGI("  Scope %zu StreamInfo[%zu]: streamIdx=%u, headNode=%lu, tailNode=%lu, nodeSize=%lu", scopeIdx, j,
            streamInfo.streamIdx, streamInfo.headNodeIdx, streamInfo.tailNodeIdx, streamInfo.nodeSize);
  }
}

void ScopeSplitPass::RequestResplit() {
  if (splitter_ != nullptr) {
    splitter_->RequestResplit();
  }
}

void ScopeSplitPass::RebuildStreamInfos(SuperKernelScopeInfo &scope) {
  std::vector<ScopeStreamInfo> newStreamInfos;
  for (const auto *node : scope.GetNodes()) {
    uint32_t streamIdx = node->GetStreamIdxInGraph();
    auto it = std::find_if(newStreamInfos.begin(), newStreamInfos.end(),
                           [streamIdx](const ScopeStreamInfo &info) { return info.streamIdx == streamIdx; });
    if (it == newStreamInfos.end()) {
      ScopeStreamInfo newInfo;
      newInfo.streamIdx = streamIdx;
      newInfo.headNodeIdx = node->GetNodeId();
      newInfo.tailNodeIdx = node->GetNodeId();
      newInfo.nodeSize = 1;
      newStreamInfos.push_back(std::move(newInfo));
    } else {
      it->tailNodeIdx = node->GetNodeId();
      it->nodeSize++;
    }
  }
  scope.SetScopeStreamInfos(std::move(newStreamInfos));
}

std::vector<uint64_t> ScopeSplitPass::GetKernelNodeIds(const SuperKernelScopeInfo &scope) {
  std::vector<uint64_t> kernelIds;
  for (const auto *node : scope.GetNodes()) {
    if (node != nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      if (node->IsScopeNode()) {
        continue;
      }
      kernelIds.push_back(node->GetNodeId());
    }
  }
  std::sort(kernelIds.begin(), kernelIds.end());
  return kernelIds;
}

// ============ InitialScopeSplitPass Implementation ============

InitialScopeSplitPass::InitialScopeSplitPass(SuperKernelGraph &inputGraph, SkHeapType heapType)
    : ScopeSplitPass(inputGraph), nodeHeap_(inputGraph, heapType) {}

void InitialScopeSplitPass::InitStreamStates() {
  SK_LOGI("[SplitScope] initializing stream states for %s", GetName().c_str());
  const auto &streams = graph_.GetStreams();
  const auto &headNodes = graph_.GetHeadNodes();

  streamStates_.clear();
  for (size_t i = 0; i < streams.size(); ++i) {
    streamStates_[i] = StreamState();
    streamStates_[i].currentNodeId = headNodes[i];
    SK_LOGI("  Stream %u: initialized with head node %lu", static_cast<uint32_t>(i), headNodes[i]);
  }
  SK_LOGI("[SplitScope] completed initializing %zu streams", streamStates_.size());
}

bool InitialScopeSplitPass::ResetStreamStates() {
  SK_LOGI("[SplitScope] resetting all stream states for next scope");

  // Save the scopeStartBreakInfo_ before reset (it was set during SkipUnfusibleNodes)
  // This will be used for the next scope's break reason if applicable

  // Clear stream-specific break info for next scope
  streamBreakInfos_.clear();
  currentScopeBreakInfo_ = ScopeBreakInfo();
  scopeStartBreakInfo_ = ScopeBreakInfo();  // Will be set again during SkipUnfusibleNodes

  for (auto &pair : streamStates_) {
    uint32_t streamIdx = pair.first;
    pair.second.isTerminated = false;
    pair.second.isSuspended = false;
    pair.second.waitingForNotify = INVALID_TASK_ID;
    SK_LOGI("  Stream %u: reset, currentNodeId=%lu", streamIdx, pair.second.currentNodeId);
  }
  SK_LOGI("[SplitScope] starting to skip unfusible nodes");
  if (!SkipUnfusibleNodes()) {
    SK_LOGE("[SplitScope] Failed to skip unfusible nodes during stream reset");
    return false;
  }
  SK_LOGI("[SplitScope] completed reset, active streams will proceed to next scope");
  return true;
}

/*!
 * \brief Skip all unfusible nodes for all streams
 *
 * This function is called at the beginning of each scope to advance each stream's
 * current position past all unfusible nodes, so that we start processing from a
 * fusible node for the next scope.
 *
 * Special handling for WAIT nodes:
 * - If the corresponding NOTIFY node has already been visited, skip the WAIT node
 * - If the NOTIFY node has not been visited, suspend the stream and keep the WAIT
 *   node as the current node (do not advance)
 *
 * Special handling for NOTIFY nodes:
 * - Always add NOTIFY nodes to visitedNotifies_ even if they are unfusible, so that
 *   dependent WAIT nodes can be properly processed
 */
bool InitialScopeSplitPass::SkipUnfusibleNodes() {
  SK_LOGI("[SkipUnfusible] starting to skip unfusible nodes for all streams");

  for (auto &pair : streamStates_) {
    uint32_t streamIdx = pair.first;
    if (!SkipUnfusibleNodesForStream(streamIdx)) {
      SK_LOGE("[SkipUnfusible] Failed to skip unfusible nodes for stream %u", streamIdx);
      return false;
    }
  }

  SK_LOGI("[SkipUnfusible] completed skipping unfusible nodes");
  return true;
}

/*!
 * \brief Skip unfusible nodes for a single stream
 *
 * \param streamIdx The stream index to process
 *
 * \return true if the stream has more nodes to process (either fusible nodes or
 *         it's suspended on an unfusible WAIT node), false if the stream is terminated
 */
bool InitialScopeSplitPass::SkipUnfusibleNodesForStream(uint32_t streamIdx) {
  StreamState &state = streamStates_[streamIdx];
  uint32_t skipCount = 0;

  if (state.currentNodeId == INVALID_TASK_ID) {
    SK_LOGI("Stream %u: Already at end (currentNodeId=INVALID), nothing to skip", streamIdx);
    return true;
  }

  while (state.currentNodeId != INVALID_TASK_ID) {
    SuperKernelBaseNode *node = graph_.GetNodeById(state.currentNodeId);
    if (node == nullptr) {
      SK_LOGE("Stream %u: node %lu not found (graph integrity error)", streamIdx, state.currentNodeId);
      return false;
    }
    if (node->IsFusible()) {
      SK_LOGI("Stream %u: Found fusible node %s, stop skipping", streamIdx, node->Format().c_str());
      break;
    }

    bool shouldContinue = ProcessUnfusibleNodeForSkip(streamIdx, node);
    if (!shouldContinue) {
      // Check if suspended (not an error) or failed (error)
      SkNodeType nodeType = node->GetNodeType();
      if (nodeType == SkNodeType::NODE_WAIT) {
        // Suspended on wait node - not an error, just stop
        return true;
      }
      // Other failures - propagate error
      return false;
    }
    state.currentNodeId = node->GetNextNodeId();
    skipCount++;
  }

  if (skipCount > 0) {
    SK_LOGI("Stream %u: Skipped %u unfusible nodes, next node=%lu", streamIdx, skipCount, state.currentNodeId);
  }
  return true;
}

/*!
 * \brief Process an unfusible node during skip phase
 * \param streamIdx The stream index
 * \param node The unfusible node to process
 * \return true if should continue skipping, false to stop
 */
bool InitialScopeSplitPass::ProcessUnfusibleNodeForSkip(uint32_t streamIdx, SuperKernelBaseNode *node) {
  SkNodeType nodeType = node->GetNodeType();

  if (nodeType == SkNodeType::NODE_WAIT) {
    bool shouldSkip = ProcessUnfusibleWaitNode(streamIdx, node);
    if (!shouldSkip) {
      SK_LOGI("Stream %u: Suspended on unfusible wait node %s", streamIdx, node->Format().c_str());
      return false;
    }
    SK_LOGI("Stream %u: Skipping unfusible wait node %s", streamIdx, node->Format().c_str());
  } else if (nodeType == SkNodeType::NODE_NOTIFY) {
    if (!HandleUnfusibleNotifyNode(node, streamIdx)) {
      SK_LOGE("Stream %u: Failed to handle unfusible notify node %s, abort processing", streamIdx,
              node->Format().c_str());
      return false;
    }
  } else {
    processedNodes_.insert(node->GetNodeId());
    SK_LOGI("Stream %u: Skipping unfusible node %s (type=%d), added to processedNodes", streamIdx,
            node->Format().c_str(), static_cast<int>(nodeType));
  }
  return true;
}

/*!
 * \brief Process an unfusible WAIT node and determine if it should be skipped
 *
 * \param streamIdx The stream index
 * \param waitNode The unfusible WAIT node to process
 *
 * \return true if the WAIT node should be skipped (advance currentNodeId),
 *         false if the stream should be suspended (keep currentNodeId)
 *
 * Processing logic:
 * 1. Find the corresponding NOTIFY node
 * 2. If NOTIFY node has already been visited: skip the WAIT node and add to processedNodes_
 * 3. If NOTIFY node has not been visited: suspend the stream and keep the WAIT node
 *    as the current node, so it will be processed again later when the NOTIFY is visited
 */
bool InitialScopeSplitPass::ProcessUnfusibleWaitNode(uint32_t streamIdx, SuperKernelBaseNode *waitNode) {
  uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();
  if (notifyId == INVALID_TASK_ID) {
    SK_LOGI("Stream %u: Wait node %s's notify node %lu not found in graph", streamIdx, waitNode->Format().c_str(),
            notifyId);
    return true;
  }
  SuperKernelBaseNode *notifyNode = graph_.GetNodeById(notifyId);

  // Error handling: notify node not found
  if (notifyNode == nullptr) {
    SK_LOGE("Stream %u: Wait node %s's notify node %lu not found", streamIdx, waitNode->Format().c_str(), notifyId);
    return true;  // Skip the wait node if notify not found to avoid hanging
  }

  // Check if notify node has been visited
  if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
    // Notify already visited, skip the wait node
    processedNodes_.insert(waitNode->GetNodeId());  // Add to processed nodes
    SK_LOGI("Stream %u: Unfusible wait node %s: notify %lu already visited, skipping and added to processedNodes_",
            streamIdx, waitNode->Format().c_str(), notifyId);
    return true;  // Should skip (advance currentNodeId)
  } else {
    // Notify not visited, suspend the stream
    StreamState &state = streamStates_[streamIdx];
    uint64_t eventId = notifyNode->GetEventId();
    state.isSuspended = true;
    state.waitingForNotify = eventId;
    SK_LOGI("Stream %u: Unfusible wait node %s: notify %lu not visited, suspending stream (eventId=0x%lx)", streamIdx,
            waitNode->Format().c_str(), notifyId, eventId);
    return false;  // Should NOT skip (keep currentNodeId for wait node)
  }
}

bool InitialScopeSplitPass::DetermineCurrentScopeBitFlags() {
  SK_LOGI("Starting to determine scope bit flags");
  uint64_t minNodeIdx = UINT64_MAX;
  SuperKernelBaseNode *minNode = nullptr;
  uint32_t activeStreams = 0;

  for (const auto &pair : streamStates_) {
    uint32_t streamIdx = pair.first;
    if (!pair.second.isTerminated && !pair.second.isSuspended && pair.second.currentNodeId != INVALID_TASK_ID) {
      activeStreams++;
      if (pair.second.currentNodeId < minNodeIdx) {
        SuperKernelBaseNode *node = graph_.GetNodeById(pair.second.currentNodeId);
        if (node != nullptr && node->IsFusible()) {
          minNodeIdx = pair.second.currentNodeId;
          minNode = node;
          SK_LOGI("Stream %u: candidate min node %s (nodeIdx=%lu)", streamIdx, node->Format().c_str(), minNodeIdx);
        }
      }
    }
  }

  if (minNode != nullptr) {
    currentScopeBitFlags_ = minNode->GetScopeBitFlags();
    std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(currentScopeBitFlags_, graph_);
    SK_LOGI("Found min node %s, scopeBitFlags=%s, scopeNames=[%s], active streams=%u", minNode->Format().c_str(),
            graph_.BitsetToString(currentScopeBitFlags_).c_str(), scopeNames.c_str(), activeStreams);
    return true;
  }

  // No fusible nodes found - log diagnostic information
  LogFusibleNodeSearchResult();

  return false;
}

void InitialScopeSplitPass::InitNodeHeap() {
  SK_LOGI("[SplitScope] initializing node heap for current scope");
  nodeHeap_.reset();
  size_t addedNodes = 0;
  for (auto &pair : streamStates_) {
    TryAddNodeToHeap(pair.first);
  }
  SK_LOGI("[SplitScope] node heap initialized with %zu nodes", nodeHeap_.size());
}

void InitialScopeSplitPass::TryAddNodeToHeap(uint32_t streamIdx) {
  StreamState &state = streamStates_[streamIdx];

  if (state.isTerminated || state.isSuspended || state.currentNodeId == INVALID_TASK_ID) {
    SK_LOGI("Stream %u: skipped (terminated=%d, suspended=%d, currentNodeId=%lu)", streamIdx, state.isTerminated,
            state.isSuspended, state.currentNodeId);
    return;
  }

  SuperKernelBaseNode *node = graph_.GetNodeById(state.currentNodeId);
  if (node == nullptr) {
    SK_LOGE("Stream %u: node %lu not found, terminating stream (graph integrity error: nodeId not in graph)", streamIdx,
            state.currentNodeId);
    state.currentNodeId = INVALID_TASK_ID;
    return;
  }

  if (processedNodes_.find(state.currentNodeId) != processedNodes_.end()) {
    SK_LOGI("Stream %u: node %lu already processed, advancing", streamIdx, state.currentNodeId);
    state.currentNodeId = node->GetNextNodeId();
    TryAddNodeToHeap(streamIdx);
    return;
  }

  // Check scopeBitFlags match
  if (node->GetScopeBitFlags() != currentScopeBitFlags_) {
    SK_LOGI("Stream %u: node %s scopeBitFlags mismatch, terminating stream", streamIdx, node->Format().c_str());
    state.isTerminated = true;
    return;
  }

  // Check fusibility
  if (!node->IsFusible()) {
    SK_LOGI("Stream %u: node %s is not fusible, terminating stream", streamIdx, node->Format().c_str());
    state.isTerminated = true;

    ScopeBreakInfo breakInfo;
    breakInfo.SetReason(ScopeBreakReason::UNFUSIBLE_NODE)
        .SetTriggerNode(node->GetNodeId(), streamIdx)
        .SetDetail("unfused node causes scope break");
    scopeSplitResultReporter_->ReportNewBreak(*currentScope_, std::move(breakInfo));

    SK_LOGI("Current scope has %zu nodes", currentScope_->GetNodes().size());
    // Record break info (overwrites previous, keeping last one)

    std::string detail = "Node " + std::to_string(node->GetNodeId()) +
                         " is unfusible: " + FusionFailReasonToStr(node->GetFusionFailReasonInfo());
    return;
  }

  // Handle Wait nodes: check if corresponding notify has been visited
  if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
    HandleWaitNode(node, streamIdx);
    // HandleWaitNode will either:
    // 1. Add wait node to heap (if notify already visited)
    // 2. Suspend the stream (if notify not yet visited)
    return;  // Don't continue with normal processing
  }

  // Add to heap (no deadlock check in this pass)
  nodeHeap_.push(state.currentNodeId);
  SK_LOGI("Stream %u: added node %s to heap, heap size=%zu", streamIdx, node->Format().c_str(), nodeHeap_.size());
}

void InitialScopeSplitPass::HandleWaitNode(SuperKernelBaseNode *waitNode, uint32_t streamIdx) {
  StreamState &state = streamStates_[streamIdx];
  uint64_t notifyId = waitNode->GetCorrespondingNotifyNodeId();

  if (notifyId == INVALID_TASK_ID) {
    nodeHeap_.push(state.currentNodeId);
    SK_LOGI("Stream %u: Wait node %s's notify node %lu not found, adding wait node to heap to avoid hanging", streamIdx,
            waitNode->Format().c_str(), notifyId);
    return;
  }

  SuperKernelBaseNode *notifyNode = graph_.GetNodeById(notifyId);
  if (notifyNode == nullptr) {
    SK_LOGE("Stream %u: Wait node %s: failed to get notify node by id %lu", streamIdx, waitNode->Format().c_str(),
            notifyId);
    return;
  }
  uint64_t eventId = notifyNode->GetEventId();
  if (visitedNotifies_.find(notifyId) != visitedNotifies_.end()) {
    // Notify already visited, add fusible wait node to heap
    nodeHeap_.push(state.currentNodeId);
    SK_LOGI("Stream %u: Wait node %s: notify %lu already visited, adding to heap", streamIdx,
            waitNode->Format().c_str(), notifyId);
  } else {
    // Suspend stream
    state.isSuspended = true;
    state.waitingForNotify = eventId;
    SK_LOGI("Stream %u: Wait node %s: notify %lu not visited, suspending stream (eventId=0x%lx)", streamIdx,
            waitNode->Format().c_str(), notifyId, eventId);
  }
}

void InitialScopeSplitPass::ProcessNotifyNode(SuperKernelBaseNode *notifyNode) {
  uint64_t eventId = notifyNode->GetEventId();
  uint64_t notifyNodeId = notifyNode->GetNodeId();
  const EventInfos *eventInfo = graph_.GetEventInfo(eventId);
  if (eventInfo == nullptr) {
    SK_LOGE("Event info for eventId=0x%lx not found (notifyNode: %s)", eventId, notifyNode->Format().c_str());
    return;
  }
  visitedNotifies_.insert(notifyNodeId);
  SK_LOGI("Notify node %s (eventId=0x%lx) added to visited, %zu wait nodes to check", notifyNode->Format().c_str(),
          eventId, eventInfo->waitNodeIdList.size());

  uint32_t resumedCount = 0;
  for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
    SuperKernelBaseNode *waitNode = graph_.GetNodeById(waitNodeId);
    if (waitNode == nullptr) {
      SK_LOGE("Wait node %lu not found (graph integrity error: eventId=0x%lx references non-existent node)", waitNodeId,
              eventId);
      continue;
    }

    uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
    StreamState &state = streamStates_[streamIdx];

    if (state.isSuspended && state.waitingForNotify == eventId) {
      state.isSuspended = false;
      state.waitingForNotify = INVALID_TASK_ID;
      TryAddNodeToHeap(streamIdx);
      resumedCount++;
      SK_LOGI("Resumed stream %u for wait node %s", streamIdx, waitNode->Format().c_str());
    }
  }
  SK_LOGI("Resumed %u suspended streams", resumedCount);
}

void InitialScopeSplitPass::ProcessResetNode(SuperKernelBaseNode *resetNode) {
  uint64_t eventId = resetNode->GetEventId();
  const EventInfos *eventInfo = graph_.GetEventInfo(eventId);
  if (eventInfo == nullptr) {
    SK_LOGE("Event info for eventId=0x%lx not found (resetNode: %s)", eventId, resetNode->Format().c_str());
    return;
  }

  if (eventInfo->notifyNodeId != INVALID_TASK_ID) {
    visitedNotifies_.erase(eventInfo->notifyNodeId);
    SK_LOGI("Erased notify %lu from visited (eventId=0x%lx)", eventInfo->notifyNodeId, eventId);
  }

  uint32_t suspendedCount = 0;
  for (uint64_t waitNodeId : eventInfo->waitNodeIdList) {
    SuperKernelBaseNode *waitNode = graph_.GetNodeById(waitNodeId);
    if (waitNode == nullptr) {
      SK_LOGE("Wait node %lu not found (graph integrity error: eventId=0x%lx references non-existent node)", waitNodeId,
              eventId);
      continue;
    }

    uint32_t streamIdx = waitNode->GetStreamIdxInGraph();
    if (streamStates_[streamIdx].waitingForNotify == eventId) {
      streamStates_[streamIdx].waitingForNotify = INVALID_TASK_ID;
      streamStates_[streamIdx].isSuspended = true;
      suspendedCount++;
      SK_LOGE("Suspended stream %u for wait node %s (eventId=0x%lx)", streamIdx, waitNode->Format().c_str(), eventId);
    }
  }
  SK_LOGE("Suspended %u streams due to reset", suspendedCount);
}

bool InitialScopeSplitPass::HandleUnfusibleNotifyNode(SuperKernelBaseNode *notifyNode, uint32_t streamIdx) {
  // Always record visited notifies for wait nodes to check
  visitedNotifies_.insert(notifyNode->GetNodeId());
  processedNodes_.insert(notifyNode->GetNodeId());
  SK_LOGI("Stream %u: Unfusible notify node %s added to visitedNotifies and processedNodes", streamIdx,
          notifyNode->Format().c_str());

  // Check if any corresponding wait nodes are suspended and need processing
  return ResumeSuspendedWaitStreams(notifyNode, streamIdx);
}

bool InitialScopeSplitPass::ResumeSuspendedWaitStreams(SuperKernelBaseNode *notifyNode, uint32_t notifyStreamIdx) {
  std::vector<uint64_t> waitNodeIds = notifyNode->GetCorrespondingWaitNodeIds();
  uint64_t eventId = notifyNode->GetEventId();

  for (uint64_t waitNodeId : waitNodeIds) {
    SuperKernelBaseNode *waitNode = graph_.GetNodeById(waitNodeId);
    if (waitNode == nullptr) {
      SK_LOGE("Stream %u: Notify node %s's wait node %lu not found", notifyStreamIdx, notifyNode->Format().c_str(),
              waitNodeId);
      return false;
    }

    uint32_t waitStreamIdx = waitNode->GetStreamIdxInGraph();
    const StreamState &waitStreamState = streamStates_[waitStreamIdx];

    // Check if the wait stream is suspended waiting for this notify's event
    if (waitStreamState.isSuspended && waitStreamState.waitingForNotify == eventId) {
      // The wait stream was suspended because this notify wasn't visited.
      // Now that we've visited the notify, we need to skip the unfusible wait node
      // and continue processing that stream.
      SK_LOGI("Stream %u: Notify node %s visited, processing suspended wait stream %u (wait node %s)", notifyStreamIdx,
              notifyNode->Format().c_str(), waitStreamIdx, waitNode->Format().c_str());

      // Resume the stream (clear suspend state)
      StreamState &mutableWaitState = streamStates_[waitStreamIdx];
      mutableWaitState.isSuspended = false;
      mutableWaitState.waitingForNotify = INVALID_TASK_ID;

      // Skip unfusible nodes on that stream (including the wait node)
      if (!SkipUnfusibleNodesForStream(waitStreamIdx)) {
        SK_LOGE("Stream %u: Failed to skip unfusible nodes for suspended wait stream %u (wait node %s)",
                notifyStreamIdx, waitStreamIdx, waitNode->Format().c_str());
        return false;
      }
    }
  }
  return true;
}

void InitialScopeSplitPass::LogFusibleNodeSearchResult() {
  SK_LOGI("[FindFusibleNodes] No fusible nodes found in any stream");

  // Check if all streams have finished (currentNodeId == INVALID_TASK_ID)
  bool allStreamsFinished = true;
  std::vector<uint32_t> nonFinishedStreamIdxs;

  for (const auto &pair : streamStates_) {
    uint32_t streamIdx = pair.first;
    const StreamState &state = pair.second;

    if (state.currentNodeId != INVALID_TASK_ID) {
      allStreamsFinished = false;
      nonFinishedStreamIdxs.push_back(streamIdx);
    }
  }

  if (allStreamsFinished) {
    SK_LOGI("[FindFusibleNodes] All streams have finished (all currentNodeId are INVALID_TASK_ID)");
  } else {
    // ERROR level when there are streams not finished but no fusible nodes found
    SK_LOGE("[FindFusibleNodes] Found %zu streams with non-INVALID currentNodeId but no fusible nodes",
            nonFinishedStreamIdxs.size());

    // Print detailed information about streams that haven't finished
    for (uint32_t streamIdx : nonFinishedStreamIdxs) {
      const StreamState &state = streamStates_.at(streamIdx);
      std::string stateInfo = state.FormatStreamStateInfo();
      SK_LOGE("[FindFusibleNodes] Stream %u state: %s", streamIdx, stateInfo.c_str());

      // If there's a current node, try to get more information
      if (state.currentNodeId != INVALID_TASK_ID) {
        SuperKernelBaseNode *node = graph_.GetNodeById(state.currentNodeId);
        if (node != nullptr) {
          bool isFusible = node->IsFusible();
          SkNodeType nodeType = node->GetNodeType();
          std::string scopeBitFlagsStr = graph_.BitsetToString(node->GetScopeBitFlags());
          SK_LOGE("[FindFusibleNodes] Stream %u current node: %s, isFusible=%d, nodeType=%d, scopeBitFlags=%s",
                  streamIdx, node->Format().c_str(), isFusible, static_cast<int>(nodeType), scopeBitFlagsStr.c_str());
        }
      }
    }
  }
}

void InitialScopeSplitPass::AddStreamInfoToScope(SuperKernelScopeInfo &scopeInfo, SuperKernelBaseNode *node) {
  uint32_t streamIdx = node->GetStreamIdxInGraph();
  auto scopeStreamInfos = scopeInfo.GetScopeStreamInfos();
  auto it = std::find_if(scopeStreamInfos.begin(), scopeStreamInfos.end(),
                         [streamIdx](const ScopeStreamInfo &info) { return info.streamIdx == streamIdx; });
  if (it == scopeStreamInfos.end()) {
    ScopeStreamInfo newInfo;
    newInfo.streamIdx = streamIdx;
    newInfo.headNodeIdx = node->GetNodeId();
    newInfo.tailNodeIdx = node->GetNodeId();
    newInfo.nodeSize = 1;
    scopeStreamInfos.push_back(std::move(newInfo));
  } else {
    it->tailNodeIdx = node->GetNodeId();
    it->nodeSize++;
  }
  scopeInfo.SetScopeStreamInfos(std::move(scopeStreamInfos));
}

bool InitialScopeSplitPass::BuildCurrentScope(SuperKernelScopeInfo &scopeInfo) {
  SK_LOGI("[SplitScope] starting to build current scope");
  scopeInfo.SetScopeBitFlags(currentScopeBitFlags_);
  currentScope_ = &scopeInfo;

  while (!nodeHeap_.empty()) {
    uint64_t nodeId = nodeHeap_.pop();

    SuperKernelBaseNode *node = graph_.GetNodeById(nodeId);
    if (node == nullptr) {
      SK_LOGE("Node %lu not found in graph (graph integrity error)", nodeId);
      return false;
    }
    if (processedNodes_.find(nodeId) != processedNodes_.end()) {
      continue;
    }

    uint32_t streamIdx = node->GetStreamIdxInGraph();

    // Add to scope
    scopeInfo.AddNode(node);
    SK_LOGI("Added node %s to scope (count=%zu)", node->Format().c_str(), scopeInfo.GetNodes().size());
    AddStreamInfoToScope(scopeInfo, node);
    processedNodes_.insert(nodeId);

    // Advance stream
    uint64_t nextNodeId = node->GetNextNodeId();
    streamStates_[streamIdx].currentNodeId = nextNodeId;

    // Handle special nodes
    if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
      ProcessNotifyNode(node);
    } else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
      ProcessResetNode(node);
    }

    // Try next node
    TryAddNodeToHeap(streamIdx);
  }

  SK_LOGI("[SplitScope] built scope with %zu nodes, %zu stream infos", scopeInfo.GetNodes().size(),
          scopeInfo.GetScopeStreamInfos().size());
  currentScope_ = nullptr;
  return true;  // Empty scope is not an error, allow it to be skipped in Run()
}

bool InitialScopeSplitPass::Run(std::vector<SuperKernelScopeInfo> &scopes) {
  SK_LOGI("[SplitScope] %s pass starting execution", GetName().c_str());

  scopes.clear();
  processedNodes_.clear();
  visitedNotifies_.clear();
  streamBreakInfos_.clear();
  currentScopeBreakInfo_ = ScopeBreakInfo();
  scopeStartBreakInfo_ = ScopeBreakInfo();

  SK_LOGI("%s: Initializing stream states", GetName().c_str());
  InitStreamStates();
  if (!SkipUnfusibleNodes()) {
    SK_LOGE("%s: Failed to skip unfusible nodes during initialization (graph integrity error)", GetName().c_str());
    return false;
  }

  size_t scopeCount = 0;
  while (true) {
    if (!DetermineCurrentScopeBitFlags()) {
      SK_LOGI("%s: No more fusible nodes, initial scope split complete", GetName().c_str());
      break;
    }

    InitNodeHeap();
    if (nodeHeap_.empty()) {
      SK_LOGI("%s: Node heap is empty, initial scope split complete", GetName().c_str());
      break;
    }

    SuperKernelScopeInfo scopeInfo;
    if (BuildCurrentScope(scopeInfo)) {
      if (!scopeInfo.GetNodes().empty()) {
        std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.GetScopeBitFlags(), graph_);
        SK_LOGI("%s: Built scope %zu with %zu nodes, %zu streams, scopeNames=[%s], breakReason=[%s]", GetName().c_str(),
                scopeCount, scopeInfo.GetNodes().size(), scopeInfo.GetScopeStreamInfos().size(), scopeNames.c_str(),
                scopeInfo.GetBreakInfo().Format().c_str());
        scopes.push_back(std::move(scopeInfo));
      } else {
        // Empty scope: skip it and continue to next iteration
        std::string scopeNames = ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.GetScopeBitFlags(), graph_);
        SK_LOGI("%s: Empty scope %zu skipped, scopeNames=[%s], continuing", GetName().c_str(), scopeCount,
                scopeNames.c_str());
      }
      scopeCount++;
    } else {
      SK_LOGE("%s: Failed to build scope %zu (graph integrity error), stopping scope split", GetName().c_str(),
              scopeCount);
      return false;
    }

    SK_LOGI("%s: Resetting stream states for next scope", GetName().c_str());
    if (!ResetStreamStates()) {
      SK_LOGE("%s: Failed to reset stream states (graph integrity error), stopping scope split", GetName().c_str());
      return false;
    }
  }

  scopeSplitResultReporter_->RestoreResplitBreakInfos(scopes);
  SK_LOGI("[SplitScopeResult] %s pass completed, total scopes generated: %zu", GetName().c_str(), scopes.size());
  PrintScopeResults(scopes, graph_, GetName().c_str());
  return true;
}

// ============ DeadlockRefinePass Implementation ============

DeadlockRefinePass::DeadlockRefinePass(SuperKernelGraph &inputGraph, SuperKernelOptionsManager &opts)
    : ScopeSplitPass(inputGraph), lockDetector_(graph_, opts) {}

bool DeadlockRefinePass::FindDeadlockInScope(const SuperKernelScopeInfo &scope, SuperKernelBaseNode **deadlockNode,
                                             SuperKernelBaseNode **deadlockWaitNode,
                                             ScopeCoreInfo *checkedScopeCoreInfo) {
  const auto &nodes = scope.GetNodes();
  SK_LOGI("[DeadlockRefine] checking scope with %zu nodes for deadlock", nodes.size());
  lockDetector_.Reset();
  lockDetector_.SetCurrentScopeId(scope.GetScopeId());

  ScopeCoreInfoCalculator coreInfoCalculator(maxCubeNum_, maxVectorNum_);
  if (!coreInfoCalculator.GetValidScopeCoreInfo(nodes)) {
    SK_LOGD("[DeadlockRefine] scope has no valid core info, skip deadlock refinement");
    *checkedScopeCoreInfo = {};
    return false;
  }
  coreInfoCalculator.Reset();

  ScopeCoreInfo candidateCoreInfo;
  ScopeCoreInfo coreInfoBeforeLastWait;

  // Track the most recent Wait node seen before each node
  SuperKernelBaseNode *lastWaitNode = nullptr;

  // Check each node for deadlock, keeping track of the nearest preceding Wait node
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto *node = nodes[i];
    if (IsExecutableKernelNode(node)) {
      coreInfoCalculator.UpdateScopeCoreInfo(*node, candidateCoreInfo);
      lockDetector_.SetScopeCoreInfo(candidateCoreInfo);
    }
    // Update lastWaitNode when we encounter a Wait node
    if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
      lastWaitNode = const_cast<SuperKernelBaseNode *>(node);
      coreInfoBeforeLastWait = candidateCoreInfo;
      SK_LOGI("Found Wait node %s at position %zu", node->Format().c_str(), i);
    }

    // Check for deadlock
    if (!lockDetector_.IsFusible(*const_cast<SuperKernelBaseNode *>(node))) {
      *deadlockNode = const_cast<SuperKernelBaseNode *>(node);
      *deadlockWaitNode = lastWaitNode;  // The nearest Wait node before deadlock point
      *checkedScopeCoreInfo = coreInfoBeforeLastWait;
      SK_LOGI("Deadlock detected at node %s (position %zu)", node->Format().c_str(), i);
      if (lastWaitNode != nullptr) {
        SK_LOGI("  Nearest Wait node before deadlock: %s", lastWaitNode->Format().c_str());
      }
      return true;
    }
    SK_LOGI("Node %s passed deadlock check", node->Format().c_str());
  }

  *checkedScopeCoreInfo = candidateCoreInfo;
  SK_LOGI("[DeadlockRefine] no deadlock found in scope");
  return false;
}

void DeadlockRefinePass::SplitScopeAtWaitNode(const SuperKernelScopeInfo &scope, SuperKernelBaseNode *waitNode,
                                              SuperKernelScopeInfo &scopeBefore, SuperKernelScopeInfo &scopeAfter) {
  scopeBefore.SetScopeBitFlags(scope.GetScopeBitFlags());
  scopeAfter.SetScopeBitFlags(scope.GetScopeBitFlags());
  scope.ClearNodeScopeIds();

  bool foundWait = false;

  for (const auto *node : scope.GetNodes()) {
    if (node == waitNode) {
      foundWait = true;
      continue;  // Don't include Wait node in either scope
    }

    if (!foundWait) {
      scopeBefore.AddNode(const_cast<SuperKernelBaseNode *>(node));
    } else {
      scopeAfter.AddNode(const_cast<SuperKernelBaseNode *>(node));
    }
  }

  RebuildStreamInfos(scopeBefore);
  RebuildStreamInfos(scopeAfter);
}

/**
 * @brief Setup break info for scopeAfter after deadlock split
 */
static void SetupScopeBeforeBreakInfo(ScopeSplitResultReporter &reporter, SuperKernelScopeInfo &scopeBefore,
                                      const ScopeBreakInfo &originalBreakInfo, SuperKernelBaseNode *deadlockNode,
                                      SuperKernelBaseNode *deadlockWaitNode, bool hasSameStructureAsOriginal) {
  if (hasSameStructureAsOriginal) {
    reporter.ReportInheritedBreak(scopeBefore, originalBreakInfo);
    SK_LOGI("[DeadlockRefine] scopeBefore structure same as original, inherits break info");
  } else {
    ScopeBreakInfo breakInfo;
    breakInfo.SetReason(ScopeBreakReason::DEADLOCK_DETECTED)
        .SetTriggerNode(deadlockNode->GetNodeId(), deadlockNode->GetStreamIdxInGraph())
        .SetDetail("Deadlock at node " + std::to_string(deadlockNode->GetNodeId()) + ", split at Wait node " +
                   std::to_string(deadlockWaitNode->GetNodeId()));
    reporter.ReportNewBreak(scopeBefore, std::move(breakInfo));
    SK_LOGI("[DeadlockRefine] scopeBefore structure different, gets deadlock break info");
  }

  SK_LOGI("[DeadlockRefine] scopeBefore break info: %s", scopeBefore.GetBreakInfo().Format().c_str());
}

static void SetupScopeAfterBreakInfo(ScopeSplitResultReporter &reporter, SuperKernelScopeInfo &scopeAfter,
                                     const ScopeBreakInfo &originalBreakInfo) {
  reporter.ReportInheritedBreak(scopeAfter, originalBreakInfo);
  SK_LOGI("[DeadlockRefine] scopeAfter inherits break info");

  SK_LOGI("[DeadlockRefine] scopeAfter break info: %s", scopeAfter.GetBreakInfo().Format().c_str());
}

/**
 * @brief Handle deadlock split logic: split scope, set break info, and manage sub-scopes
 * @param workingScope The scope to split
 * @param deadlockNode The node where deadlock is detected
 * @param deadlockWaitNode The wait node to split at
 * @param outputScopes Output scopes list to store valid scopes
 * @param pendingScope Output pending scope for further processing
 * @return Processing result of deadlock resolution
 */
ScopeProcessResult DeadlockRefinePass::HandleDeadlockSplit(SuperKernelScopeInfo &workingScope,
                                                           SuperKernelBaseNode *deadlockNode,
                                                           SuperKernelBaseNode *deadlockWaitNode,
                                                           const ScopeCoreInfo &scopeBeforeCoreInfo,
                                                           std::vector<SuperKernelScopeInfo> &outputScopes,
                                                           std::optional<SuperKernelScopeInfo> &pendingScope) {
  // Save original scope break information
  const ScopeBreakInfo &originalBreakInfo = workingScope.GetBreakInfo();
  const uint16_t originalScopeId = workingScope.GetScopeId();

  // Split the scope at the target wait node
  SuperKernelScopeInfo scopeBefore;
  SuperKernelScopeInfo scopeAfter;
  SplitScopeAtWaitNode(workingScope, deadlockWaitNode, scopeBefore, scopeAfter);
  lockDetector_.RemoveScopeInfo(originalScopeId);
  RecordDeadlockSplitResult(deadlockNode, deadlockWaitNode);
  deadlockNode->SetFusionFailReason(FusionFailReason::EXIST_DEADLOCK);

  SK_LOGI("[DeadlockRefine] Deadlock detected at node %s, splitting at Wait node %s", deadlockNode->Format().c_str(),
          deadlockWaitNode->Format().c_str());
  SK_LOGI("[DeadlockRefine] Before split: original=%zu nodes, scopeBefore=%zu, scopeAfter=%zu",
          workingScope.GetNodes().size(), scopeBefore.GetNodes().size(), scopeAfter.GetNodes().size());

  if (scopeSplitResultReporter_ != nullptr) {
    bool hasSameStructureAsOriginal = scopeSplitResultReporter_->HasSameScopeStructure(workingScope, scopeBefore);
    SetupScopeBeforeBreakInfo(*scopeSplitResultReporter_, scopeBefore, originalBreakInfo, deadlockNode,
                              deadlockWaitNode, hasSameStructureAsOriginal);
    SetupScopeAfterBreakInfo(*scopeSplitResultReporter_, scopeAfter, originalBreakInfo);
  }
  // Add valid scopeBefore to output
  if (!scopeBefore.GetNodes().empty()) {
    if (scopeBeforeCoreInfo.IsValid()) {
      scopeBefore.SetScopeCoreInfo(scopeBeforeCoreInfo);
      lockDetector_.UpdateScopeInfo(scopeBefore);
    }
    lockDetector_.SetNotifyNodesExpandNumForScope(scopeBefore, scopeBeforeCoreInfo);
    outputScopes.push_back(std::move(scopeBefore));
  } else {
    SK_LOGI("[DeadlockRefine] scopeBefore is empty after split, no scope added before Wait node");
  }

  // Set pending scope for further processing
  if (!scopeAfter.GetNodes().empty()) {
    pendingScope = std::move(scopeAfter);
    SK_LOGI("[DeadlockRefine] scopeAfter has %zu nodes and will be processed further", pendingScope->GetNodes().size());
  } else {
    SK_LOGI("[DeadlockRefine] scopeAfter is empty, no further processing required for this scope");
  }

  return ScopeProcessResult::DEADLOCK_RESOLVED;
}

ScopeProcessResult DeadlockRefinePass::ProcessSingleScope(SuperKernelScopeInfo &&scopeToProcess,
                                                          std::vector<SuperKernelScopeInfo> &outputScopes,
                                                          std::optional<SuperKernelScopeInfo> &pendingScope) {
  pendingScope.reset();

  SuperKernelScopeInfo workingScope = std::move(scopeToProcess);
  workingScope.ClearScopeCoreInfo();
  if (workingScope.GetNodes().empty()) {
    SK_LOGI("[DeadlockRefine] ProcessSingleScope called with empty scope, nothing to do");
    return ScopeProcessResult::NO_DEADLOCK;
  }
  SuperKernelBaseNode *deadlockNode = nullptr;
  SuperKernelBaseNode *deadlockWaitNode = nullptr;
  ScopeCoreInfo checkedScopeCoreInfo;

  if (!FindDeadlockInScope(workingScope, &deadlockNode, &deadlockWaitNode, &checkedScopeCoreInfo)) {
    if (checkedScopeCoreInfo.IsValid()) {
      workingScope.SetScopeCoreInfo(checkedScopeCoreInfo);
      lockDetector_.UpdateScopeInfo(workingScope);
      lockDetector_.SetNotifyNodesExpandNumForScope(workingScope, checkedScopeCoreInfo);
      SK_LOGI("[DeadlockRefine] Scope has no deadlock, added as a whole with %zu nodes",
              workingScope.GetNodes().size());
    } else {
      lockDetector_.RemoveScopeInfo(workingScope.GetScopeId());
      SK_LOGD(
          "[DeadlockRefine] Deadlock refinement skipped because SK core info is unavailable, scope kept with "
          "%zu nodes",
          workingScope.GetNodes().size());
    }
    outputScopes.push_back(std::move(workingScope));
    return ScopeProcessResult::NO_DEADLOCK;
  }

  // Deadlock found but no suitable Wait node to split at: treat as fatal
  if (deadlockWaitNode == nullptr) {
    SK_LOGE("[DeadlockRefine] Deadlock at node %s but no Wait node found to split; refinement failed",
            deadlockNode != nullptr ? deadlockNode->Format().c_str() : "<null>");
    return ScopeProcessResult::DEADLOCK_UNRESOLVED;
  }

  // Execute deadlock split logic
  return HandleDeadlockSplit(workingScope, deadlockNode, deadlockWaitNode, checkedScopeCoreInfo, outputScopes,
                             pendingScope);
}

bool DeadlockRefinePass::Run(std::vector<SuperKernelScopeInfo> &scopes) {
  SK_LOGI("[DeadlockRefine] %s pass starting execution", GetName().c_str());
  SK_LOGI("[DeadlockRefine] input scopes count: %zu", scopes.size());

  std::vector<SuperKernelScopeInfo> refinedScopes;
  size_t splitCount = 0;

  if (!scopes.empty() && GetDeviceCoreNums(maxCubeNum_, maxVectorNum_) != ACL_SUCCESS) {
    scopes.clear();
    return false;
  }

  lockDetector_.SetScopeInfos(scopes);

  for (size_t i = 0; i < scopes.size(); ++i) {
    SK_LOGI("[DeadlockRefine] Processing scope index %zu with %zu nodes", i, scopes[i].GetNodes().size());

    SuperKernelScopeInfo currentScope = std::move(scopes[i]);
    std::optional<SuperKernelScopeInfo> pendingScope;

    // Repeatedly process currentScope; any remaining part after a split is
    // returned via pendingScope and processed in the next iteration. This
    // ensures that for an original scope that is split into multiple
    // pieces, the resulting scopes are emitted in order and remain
    // contiguous in refinedScopes.
    while (true) {
      ScopeProcessResult result = ProcessSingleScope(std::move(currentScope), refinedScopes, pendingScope);

      if (result == ScopeProcessResult::DEADLOCK_UNRESOLVED) {
        SK_LOGE("[DeadlockRefine] Scope %zu: deadlock detected and cannot be resolved, aborting refinement", i);
        scopes.clear();
        return false;
      }

      if (result == ScopeProcessResult::DEADLOCK_RESOLVED) {
        splitCount++;
      }

      if (!pendingScope.has_value()) {
        // No remaining part to process for this original scope
        break;
      }

      currentScope = std::move(*pendingScope);
      pendingScope.reset();
    }
  }

  scopes = std::move(refinedScopes);

  // Reset notify expand numbers for all scopes to ensure pass is reentrant
  for (auto &scope : scopes) {
    lockDetector_.ResetNotifyExpandNumForScope(scope);
  }

  // Reset visited state to ensure pass is reentrant
  lockDetector_.Reset();

  SK_LOGI("[DeadlockRefine] %s pass completed, split %zu scopes, total scopes: %zu", GetName().c_str(), splitCount,
          scopes.size());
  PrintScopeResults(scopes, graph_, GetName().c_str());
  return true;
}

// ============ ScheModeKernelSplitPass Implementation ============

ScheModeKernelSplitPass::ScheModeKernelSplitPass(SuperKernelGraph &inputGraph) : ScopeSplitPass(inputGraph) {}

void ScheModeKernelSplitPass::SplitScopeAtNode(const SuperKernelScopeInfo &scope, SuperKernelBaseNode *splitNode,
                                               SuperKernelScopeInfo &scopeBefore, SuperKernelScopeInfo &scopeAfter) {
  scopeBefore.SetScopeBitFlags(scope.GetScopeBitFlags());
  scopeAfter.SetScopeBitFlags(scope.GetScopeBitFlags());
  scope.ClearNodeScopeIds();

  bool foundSplit = false;
  for (const auto *node : scope.GetNodes()) {
    if (node == splitNode) {
      foundSplit = true;
    }
    if (!foundSplit) {
      scopeBefore.AddNode(const_cast<SuperKernelBaseNode *>(node));
    } else {
      scopeAfter.AddNode(const_cast<SuperKernelBaseNode *>(node));
    }
  }

  RebuildStreamInfos(scopeBefore);
  RebuildStreamInfos(scopeAfter);
}

static void SetupScheModeScopeBeforeBreakInfo(ScopeSplitResultReporter &reporter, SuperKernelScopeInfo &scopeBefore,
                                              const ScopeBreakInfo &originalBreakInfo, SuperKernelBaseNode *splitNode,
                                              const std::string &coreMismatchDetail, bool hasSameStructureAsOriginal) {
  if (hasSameStructureAsOriginal) {
    reporter.ReportInheritedBreak(scopeBefore, originalBreakInfo);
    SK_LOGI("[ScheModeSplit] scopeBefore structure same as original, inherits break info");
  } else {
    std::vector<uint64_t> syncAllNodeIds;
    for (const auto *node : scopeBefore.GetNodes()) {
      if (IsExecutableKernelNode(node) && node->RequiresExactCoreMatch()) {
        syncAllNodeIds.push_back(node->GetNodeId());
      }
    }
    ScopeBreakInfo breakInfo;
    breakInfo.SetReason(ScopeBreakReason::SYNCALL_OP_DROP)
        .SetTriggerNode(splitNode->GetNodeId(), splitNode->GetStreamIdxInGraph())
        .SetSyncAllNodeIds(std::move(syncAllNodeIds))
        .SetDetail(coreMismatchDetail);
    reporter.ReportNewBreak(scopeBefore, std::move(breakInfo));
    SK_LOGI("[ScheModeSplit] scopeBefore structure different, set ScheMode break info");
  }

  SK_LOGI("[ScheModeSplit] scopeBefore break info: %s", scopeBefore.GetBreakInfo().Format().c_str());
}

std::string ScheModeKernelSplitPass::BuildScheModeCoreMismatchDetail(const ScopeCoreInfo &scopeBeforeCoreInfo,
                                                                     const ScopeCoreInfo &candidateScopeCoreInfo,
                                                                     const SuperKernelScopeInfo &scopeBefore,
                                                                     const SuperKernelBaseNode &triggerNode) const {
  const auto formatScopeCoreInfo = [](const ScopeCoreInfo &coreInfo) {
    return std::string(to_string(coreInfo.type)) + "(blocks=" + std::to_string(coreInfo.numBlocks) +
           ",C=" + std::to_string(coreInfo.GetCubeNum()) + ",V=" + std::to_string(coreInfo.GetVectorNum()) + ")";
  };
  const uint32_t candidateCubeNum = candidateScopeCoreInfo.GetCubeNum();
  const uint32_t candidateVectorNum = candidateScopeCoreInfo.GetVectorNum();
  std::string mismatch;

  const auto checkExactMismatch = [&](const SuperKernelBaseNode &node) {
    if (!mismatch.empty() || !node.RequiresExactCoreMatch()) {
      return;
    }
    const bool vectorOnly = node.GetCubeNum() == 0 && node.GetVecNum() != 0;
    const uint32_t expected = vectorOnly ? node.GetVecNum() : node.GetCubeNum();
    const uint32_t actual = vectorOnly ? candidateVectorNum : candidateCubeNum;
    if (expected != actual) {
      mismatch = std::string("ScheMode ") + (vectorOnly ? "V" : "C") + "(expected=" + std::to_string(expected) +
                 ",actual=" + std::to_string(actual) + ")";
    }
  };

  checkExactMismatch(triggerNode);
  for (const auto *node : scopeBefore.GetNodes()) {
    if (IsExecutableKernelNode(node)) {
      checkExactMismatch(*node);
    }
  }
  if (mismatch.empty()) {
    mismatch = "no candidate satisfies current scope requirements";
  }

  return "beforeSk=" + formatScopeCoreInfo(scopeBeforeCoreInfo) +
         ", candidateSk=" + formatScopeCoreInfo(candidateScopeCoreInfo) + ", mismatch=" + mismatch;
}

ScheModeScopeProcessResult ScheModeKernelSplitPass::ProcessSingleScope(
    SuperKernelScopeInfo &&scopeToProcess, std::vector<SuperKernelScopeInfo> &outputScopes,
    std::optional<SuperKernelScopeInfo> &pendingScope) {
  pendingScope.reset();

  SuperKernelScopeInfo workingScope = std::move(scopeToProcess);
  workingScope.ClearScopeCoreInfo();
  if (workingScope.GetNodes().empty()) {
    SK_LOGI("[ScheModeSplit] ProcessSingleScope called with empty scope, nothing to do");
    return ScheModeScopeProcessResult::NO_SPLIT;
  }

  ScopeCoreInfoCalculator coreInfoCalculator(maxCubeNum_, maxVectorNum_);
  ScopeCoreInfo validScopeCoreInfo;
  if (coreInfoCalculator.GetValidScopeCoreInfo(workingScope.GetNodes(), validScopeCoreInfo)) {
    workingScope.SetScopeCoreInfo(validScopeCoreInfo);
    outputScopes.push_back(std::move(workingScope));
    return ScheModeScopeProcessResult::NO_SPLIT;
  }
  coreInfoCalculator.Reset();

  ScopeCoreInfo currentCoreInfo;

  auto &nodes = workingScope.GetNodes();
  for (auto *node : nodes) {
    if (!IsExecutableKernelNode(node)) {
      continue;
    }

    ScopeCoreInfo candidateCoreInfo;
    if (!coreInfoCalculator.UpdateScopeCoreInfo(*node, candidateCoreInfo)) {
      // Save original scope's break info before split
      const ScopeBreakInfo &originalBreakInfo = workingScope.GetBreakInfo();
      SuperKernelScopeInfo scopeBefore;
      SuperKernelScopeInfo scopeAfter;
      SplitScopeAtNode(workingScope, node, scopeBefore, scopeAfter);
      RecordScheModeSplitResult(node);

      const std::string coreMismatchDetail =
          BuildScheModeCoreMismatchDetail(currentCoreInfo, candidateCoreInfo, scopeBefore, *node);
      if (scopeSplitResultReporter_ != nullptr) {
        bool hasSameStructureAsOriginal = scopeSplitResultReporter_->HasSameScopeStructure(workingScope, scopeBefore);
        SetupScheModeScopeBeforeBreakInfo(*scopeSplitResultReporter_, scopeBefore, originalBreakInfo, node,
                                          coreMismatchDetail, hasSameStructureAsOriginal);
        SetupScopeAfterBreakInfo(*scopeSplitResultReporter_, scopeAfter, originalBreakInfo);
      }

      SK_LOGI("[ScheModeSplit] split before kernel %lu because no SK core candidate remains", node->GetNodeId());

      if (!scopeBefore.GetNodes().empty()) {
        if (currentCoreInfo.IsValid()) {
          scopeBefore.SetScopeCoreInfo(currentCoreInfo);
        }
        outputScopes.push_back(std::move(scopeBefore));
      }
      if (!scopeAfter.GetNodes().empty()) {
        pendingScope = std::move(scopeAfter);
      }
      return ScheModeScopeProcessResult::SPLIT_RESOLVED;
    }
    currentCoreInfo = candidateCoreInfo;
  }

  if (currentCoreInfo.IsValid()) {
    workingScope.SetScopeCoreInfo(currentCoreInfo);
  }
  outputScopes.push_back(std::move(workingScope));
  return ScheModeScopeProcessResult::NO_SPLIT;
}

bool ScheModeKernelSplitPass::Run(std::vector<SuperKernelScopeInfo> &scopes) {
  SK_LOGI("[ScheModeSplit] %s pass starting execution", GetName().c_str());
  SK_LOGI("[ScheModeSplit] input scopes count: %zu", scopes.size());

  std::vector<SuperKernelScopeInfo> resScopes;
  size_t splitCount = 0;

  if (scopes.empty()) {
    return true;
  }
  if (GetDeviceCoreNums(maxCubeNum_, maxVectorNum_) != ACL_SUCCESS) {
    scopes.clear();
    return false;
  }

  for (size_t i = 0; i < scopes.size(); ++i) {
    SK_LOGI("[ScheModeSplit] Processing scope index %zu with %zu nodes", i, scopes[i].GetNodes().size());
    SuperKernelScopeInfo currentScope = std::move(scopes[i]);
    std::optional<SuperKernelScopeInfo> pendingScope;

    while (true) {
      ScheModeScopeProcessResult result = ProcessSingleScope(std::move(currentScope), resScopes, pendingScope);
      if (result == ScheModeScopeProcessResult::SPLIT_RESOLVED) {
        splitCount++;
      }

      if (!pendingScope.has_value()) {
        break;
      }

      currentScope = std::move(*pendingScope);
      pendingScope.reset();
    }
  }

  scopes = std::move(resScopes);
  SK_LOGI("[ScheModeSplit] %s pass completed, split %zu scopes, total scopes: %zu", GetName().c_str(), splitCount,
          scopes.size());
  PrintScopeResults(scopes, graph_, GetName().c_str());
  return true;
}

// ============ SuperKernelScopeSplitter Implementation ============

SuperKernelScopeSplitter::SuperKernelScopeSplitter(SuperKernelGraph &inputGraph, SuperKernelOptionsManager &opts)
    : graph_(inputGraph), opts_(&opts) {
  const auto *option =
      static_cast<const AggressiveOptStrategiesOption *>(opts.GetOption(aclskOptionType::AGGRESSIVE_OPT_STRATEGIES));
  enableTaskBreakerBypass_ = (option != nullptr && option->GetValue().taskBreakerBypass == 1);

  auto autoOpParallel = opts.GetOption(aclskOptionType::AUTO_OP_PARALLEL);
  SkHeapType heapType = SkHeapType::PRIORITY_QUEUE;
  if (autoOpParallel != nullptr) {
    heapType = static_cast<SkHeapType>(autoOpParallel->GetIntValue());
  }

  auto perOpMaxCoreOpt = opts.GetOption(aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM);
  enablePerOpMaxCore_ = (perOpMaxCoreOpt != nullptr && perOpMaxCoreOpt->GetIntValue() == 1);

  if (enablePerOpMaxCore_) {
    SK_LOGI("[ScopeSplitter] DEBUG_PER_OP_MAX_CORE_NUM enabled");
    passes_.push_back(std::make_unique<PerOpMaxCoreSplitPass>(inputGraph));
  } else {
    if (enableTaskBreakerBypass_) {
      SK_LOGI("[ScopeSplitter] taskBreakerBypass enabled");
    }
    passes_.push_back(std::make_unique<InitialScopeSplitPass>(inputGraph, heapType));
    passes_.push_back(std::make_unique<ScheModeKernelSplitPass>(inputGraph));
    passes_.push_back(std::make_unique<DeadlockRefinePass>(inputGraph, *opts_));
    if (enableTaskBreakerBypass_) {
      passes_.push_back(std::make_unique<DefaultNodeProcessPass>(inputGraph));
      passes_.push_back(std::make_unique<DeadlockRefinePass>(inputGraph, *opts_));
    }
    passes_.push_back(std::make_unique<EventOnlyStreamRemovePass>(inputGraph));
  }

  for (auto &pass : passes_) {
    pass->SetSplitter(this);
    pass->SetScopeSplitResultReporter(&splitResultReporter_);
  }
}

void SuperKernelScopeSplitter::InitDefaultNodeFusibility() {
  if (!enableTaskBreakerBypass_) {
    return;
  }
  SK_LOGI("[ScopeSplitter] Initializing default node fusibility for taskBreakerBypass");
  uint32_t count = 0;
  const auto &headNodes = graph_.GetHeadNodes();
  for (uint64_t headNodeId : headNodes) {
    uint64_t nodeId = headNodeId;
    while (nodeId != INVALID_TASK_ID) {
      SuperKernelBaseNode *node = graph_.GetNodeById(nodeId);
      if (node == nullptr) {
        break;
      }
      if (node->GetNodeType() == SkNodeType::NODE_DEFAULT) {
        node->SetIsFusible(true);
        count++;
        SK_LOGI("[ScopeSplitter] Default node %s set isFusible=true", node->Format().c_str());
      }
      nodeId = node->GetNextNodeId();
    }
  }
  SK_LOGI("[ScopeSplitter] Initialized %u default nodes as fusible", count);
}

bool SuperKernelScopeSplitter::FinalizeScopeCoreInfo() {
  if (scopeInfos_.empty()) {
    return true;
  }

  int64_t maxDeviceCubeNum = 0;
  int64_t maxDeviceVectorNum = 0;
  if (GetDeviceCoreNums(maxDeviceCubeNum, maxDeviceVectorNum) != ACL_SUCCESS) {
    return false;
  }

  ScopeCoreInfoCalculator coreInfoCalculator(maxDeviceCubeNum, maxDeviceVectorNum);
  for (auto &scope : scopeInfos_) {
    scope.ClearScopeCoreInfo();
    if (std::none_of(scope.GetNodes().begin(), scope.GetNodes().end(), IsExecutableKernelNode)) {
      SK_LOGD("[ScopeCoreFinalize] scope %u has no executable kernel, skip finalization", scope.GetScopeId());
      continue;
    }
    ScopeCoreInfo scopeCoreInfo;
    if (!coreInfoCalculator.GetValidScopeCoreInfo(scope.GetNodes(), scopeCoreInfo)) {
      SK_LOGE("[ScopeCoreFinalize] scope %u has no valid final SK core info", scope.GetScopeId());
      return false;
    }
    scope.SetScopeCoreInfo(scopeCoreInfo);
    SK_LOGI("[ScopeCoreFinalize] scope %u final core: type=%s, blocks=%u, cube=%u, vector=%u", scope.GetScopeId(),
            to_string(scopeCoreInfo.type), scopeCoreInfo.numBlocks, scopeCoreInfo.GetCubeNum(),
            scopeCoreInfo.GetVectorNum());
  }
  return true;
}

bool SuperKernelScopeSplitter::SplitGraph() {
  SK_LOGI("[ScopeSplitPipeline] starting scope splitting pipeline");
  InitDefaultNodeFusibility();
  splitResultReporter_.Reset();
  scopeInfos_.clear();
  needResplit_ = false;

  const uint32_t maxIterations = 10;
  uint32_t iteration = 0;

  do {
    iteration++;
    needResplit_ = false;
    scopeInfos_.clear();

    SK_LOGI("[ScopeSplitPipeline] iteration %u", iteration);
    {
      SK_LOG_CONTEXT_SIMPLE("sk_scope_split.log");
      SK_LOGI("==========[ScopeSplitPipeline] iteration %u==========", iteration);
    }

    for (auto &pass : passes_) {
      SK_LOGI("[ScopeSplitPipeline] running pass: %s", pass->GetName().c_str());
      {
        SK_LOG_CONTEXT_SIMPLE("sk_scope_split.log");
        SK_LOGI("===========[ScopeSplitPipeline] running pass: %s==========", pass->GetName().c_str());
      }
      if (!pass->Run(scopeInfos_)) {
        SK_LOGE("[ScopeSplitPipeline] pass %s failed (input scopes: %zu)", pass->GetName().c_str(), scopeInfos_.size());
        return false;
      }

      if (needResplit_) {
        SK_LOGI("[ScopeSplitPipeline] re-split requested by %s, starting new iteration", pass->GetName().c_str());
        break;
      }
    }

    if (iteration >= maxIterations) {
      SK_LOGW("[ScopeSplitPipeline] reached maximum iterations (%u), stopping", maxIterations);
      break;
    }
  } while (needResplit_);

  if (!enablePerOpMaxCore_ && !FinalizeScopeCoreInfo()) {
    SK_LOGE("[ScopeSplitPipeline] failed to finalize SK core info");
    scopeInfos_.clear();
    return false;
  }

  ScopeSplitPass::PrintScopeResults(scopeInfos_, graph_, "ScopeSplitPipelineFinal");

  // Print Scope Break Reason Report
  PrintScopeBreakReasonReport();

  return true;
}

// ============ DefaultNodeProcessPass Implementation ============

DefaultNodeProcessPass::DefaultNodeProcessPass(SuperKernelGraph &inputGraph) : ScopeSplitPass(inputGraph) {}

std::unordered_map<uint32_t, StreamDefaultInfo> DefaultNodeProcessPass::CollectStreamInfo(
    const SuperKernelScopeInfo &scope) {
  std::unordered_map<uint32_t, StreamDefaultInfo> streamInfoMap;
  for (const auto *node : scope.GetNodes()) {
    uint32_t streamIdx = node->GetStreamIdxInGraph();
    auto nodeType = node->GetNodeType();
    if (nodeType == SkNodeType::NODE_KERNEL) {
      streamInfoMap[streamIdx].hasKernel = true;
    } else if (nodeType == SkNodeType::NODE_DEFAULT) {
      streamInfoMap[streamIdx].defaults.push_back(const_cast<SuperKernelBaseNode *>(node));
    }
  }
  return streamInfoMap;
}

void DefaultNodeProcessPass::MarkDefaultsUnfusible(const std::vector<SuperKernelBaseNode *> &defaultNodes) {
  for (auto *node : defaultNodes) {
    node->SetIsFusible(false);
    node->SetFusionFailReason(FusionFailReason::DEFAULT_NODE);
    SK_LOGI("[DefaultNodeProcess] Default %s marked unfusible (stream has kernel)", node->Format().c_str());
  }
}

void DefaultNodeProcessPass::RemoveDefaultsAndStreams(SuperKernelScopeInfo &scope,
                                                      const std::vector<SuperKernelBaseNode *> &defaultsToRemove,
                                                      const std::unordered_set<uint32_t> &streamsToRemove) {
  auto nodes = scope.GetNodes();
  std::vector<SuperKernelBaseNode *> remaining;
  for (auto *node : nodes) {
    if (node->GetNodeType() == SkNodeType::NODE_DEFAULT) {
      SK_LOGI("[DefaultNodeProcess] Removing default: %s", node->Format().c_str());
      continue;
    }
    remaining.push_back(node);
  }
  scope.SetNodes(std::move(remaining));

  for (uint32_t streamIdx : streamsToRemove) {
    RemoveStreamFromScope(scope, streamIdx);
  }
  RebuildStreamInfos(scope);
}

void DefaultNodeProcessPass::RemoveStreamFromScope(SuperKernelScopeInfo &scope, uint32_t streamIdx) {
  auto nodes = scope.GetNodes();
  std::vector<SuperKernelBaseNode *> remaining;
  for (auto *node : nodes) {
    if (node->GetStreamIdxInGraph() != streamIdx) {
      remaining.push_back(node);
    } else {
      SK_LOGI("[DefaultNodeProcess] Removing event node: %s (stream %u)", node->Format().c_str(), streamIdx);
    }
  }
  scope.SetNodes(std::move(remaining));
}

uint32_t DefaultNodeProcessPass::ProcessSingleScope(SuperKernelScopeInfo &scope) {
  auto streamInfoMap = CollectStreamInfo(scope);

  std::vector<SuperKernelBaseNode *> defaultsToMark;
  std::vector<SuperKernelBaseNode *> defaultsToRemove;
  std::unordered_set<uint32_t> streamsToRemove;

  for (const auto &[streamIdx, info] : streamInfoMap) {
    if (info.defaults.empty()) {
      continue;
    }
    if (info.hasKernel) {
      defaultsToMark.insert(defaultsToMark.end(), info.defaults.begin(), info.defaults.end());
    } else {
      defaultsToRemove.insert(defaultsToRemove.end(), info.defaults.begin(), info.defaults.end());
      streamsToRemove.insert(streamIdx);
    }
  }

  if (defaultsToMark.empty() && defaultsToRemove.empty()) {
    return 0;
  }

  SK_LOGI("[DefaultNodeProcess] Scope has %zu defaults (%zu to mark, %zu to remove)",
          defaultsToMark.size() + defaultsToRemove.size(), defaultsToMark.size(), defaultsToRemove.size());

  if (!defaultsToMark.empty()) {
    MarkDefaultsUnfusible(defaultsToMark);
    SK_LOGI("[DefaultNodeProcess] %zu defaults marked unfusible", defaultsToMark.size());
    return static_cast<uint32_t>(defaultsToMark.size());
  }

  RemoveDefaultsAndStreams(scope, defaultsToRemove, streamsToRemove);
  SK_LOGI("[DefaultNodeProcess] %zu defaults removed", defaultsToRemove.size());
  return 0;
}

bool DefaultNodeProcessPass::Run(std::vector<SuperKernelScopeInfo> &scopes) {
  SK_LOGI("[DefaultNodeProcess] Run starting, scopes: %zu", scopes.size());
  scopeSplitResultReporter_->CaptureResplitScopes(scopes);
  uint32_t totalMarked = 0;
  for (size_t i = 0; i < scopes.size(); ++i) {
    uint32_t marked = ProcessSingleScope(scopes[i]);
    totalMarked += marked;
  }
  if (totalMarked > 0) {
    SK_LOGI("[DefaultNodeProcess] %u defaults marked unfusible, triggering resplit", totalMarked);
    scopes.clear();
    RequestResplit();
    return true;
  }
  std::vector<SuperKernelScopeInfo> valid;
  for (size_t i = 0; i < scopes.size(); ++i) {
    if (!scopes[i].GetNodes().empty()) {
      valid.push_back(std::move(scopes[i]));
    } else {
      SK_LOGI("[DefaultNodeProcess] Scope %zu empty after removal, dropping", i);
    }
  }
  scopes = std::move(valid);
  SK_LOGI("[DefaultNodeProcess] Run completed, scopes: %zu", scopes.size());
  PrintScopeResults(scopes, graph_, GetName().c_str());
  return true;
}

// ============ PrintScopeBreakReasonReport Helper Functions ============

std::string FormatNodeIds(const SuperKernelScopeInfo &scope) {
  std::string result;
  for (const auto *node : scope.GetNodes()) {
    if (node != nullptr) {
      if (!result.empty()) result += ", ";
      result += std::to_string(node->GetNodeId());
    }
  }
  return result;
}

std::string FormatKernelIds(const std::vector<uint64_t> &kernelIds, size_t maxDisplay = SIZE_MAX) {
  std::string result;
  size_t limit = (maxDisplay == SIZE_MAX) ? kernelIds.size() : maxDisplay;
  for (size_t j = 0; j < kernelIds.size() && j < limit; ++j) {
    if (j > 0) result += ", ";
    result += std::to_string(kernelIds[j]);
  }
  if (kernelIds.size() > limit) result += "...";
  return result;
}

std::unordered_map<uint16_t, size_t> BuildScopeIdIndexMap(const std::vector<SuperKernelScopeInfo> &scopeInfos) {
  std::unordered_map<uint16_t, size_t> map;
  for (size_t i = 0; i < scopeInfos.size(); ++i) {
    map[scopeInfos[i].GetScopeId()] = i;
  }
  return map;
}

void PrintScopeSummary(const SuperKernelScopeInfo &scope, size_t idx) {
  SK_LOGI("Scope %zu (scopeId=%u): %zu nodes", idx, scope.GetScopeId(), scope.GetNodes().size());
}

void PrintTriggerNode(const ScopeBreakInfo &breakInfo, SuperKernelGraph &graph) {
  if (breakInfo.GetTriggerNodeId() == INVALID_TASK_ID) return;
  SuperKernelBaseNode *node = graph.GetNodeById(breakInfo.GetTriggerNodeId());
  if (node != nullptr) {
    SK_LOGI("  Trigger node: %s", node->Format().c_str());
  }
}

void PrintKernelNodes(const std::vector<uint64_t> &kernelIds) {
  if (kernelIds.empty()) return;
  SK_LOGI("  Kernel nodes: %zu (ids: %s)", kernelIds.size(), FormatKernelIds(kernelIds, 5).c_str());
}

std::unordered_map<ScopeBreakReason, size_t> CountRootBreakReasons(
    const std::vector<SuperKernelScopeInfo> &scopeInfos, const std::unordered_map<uint16_t, size_t> &scopeIdToIdx) {
  std::unordered_map<ScopeBreakReason, size_t> counts;
  for (const auto &scope : scopeInfos) {
    ScopeBreakInfo rootInfo = FindRootBreakInfo(scope, scopeIdToIdx, scopeInfos);
    counts[rootInfo.GetReason()]++;
  }
  return counts;
}

void PrintRootBreakReasonSummary(const std::unordered_map<ScopeBreakReason, size_t> &rootReasonCounts) {
  SK_LOGI("Root Break Reason Summary:");
  for (const auto &pair : rootReasonCounts) {
    SK_LOGI("  %s: %zu scopes", to_string(pair.first), pair.second);
  }
}
ScopeBreakInfo FindRootBreakInfo(const SuperKernelScopeInfo &scope,
                                 const std::unordered_map<uint16_t, size_t> &scopeIdToIdx,
                                 const std::vector<SuperKernelScopeInfo> &scopeInfos) {
  const SuperKernelScopeInfo *currentScope = &scope;
  while (currentScope->GetBreakInfo().GetParentScopeId() != INVALID_SCOPE_ID) {
    auto it = scopeIdToIdx.find(currentScope->GetBreakInfo().GetParentScopeId());
    if (it != scopeIdToIdx.end()) {
      currentScope = &scopeInfos[it->second];
    } else {
      break;
    }
  }
  return currentScope->GetBreakInfo();
}

void SuperKernelScopeSplitter::PrintAllScopesDetail(const std::vector<SuperKernelScopeInfo> &scopeInfos) {
  SK_LOGI("All Scopes Detail");
  for (size_t i = 0; i < scopeInfos.size(); ++i) {
    const auto &scope = scopeInfos[i];
    std::string nodeIdsStr = FormatNodeIds(scope);
    std::string kernelIdsStr = FormatKernelIds(ScopeSplitPass::GetKernelNodeIds(scope));
    SK_LOGI("  [%zu] scopeId=%u, allNodeIds=[%s], kernelIds=[%s], breakReason=[%s]", i, scope.GetScopeId(),
            nodeIdsStr.c_str(), kernelIdsStr.c_str(), scope.GetBreakInfo().Format().c_str());
  }
}

void SuperKernelScopeSplitter::PrintScopeBreakReasonReport() {
  SK_LOGI("Scope Break Reason Report");
  SK_LOGI("Total scopes: %zu", scopeInfos_.size());

  auto scopeIdToIdx = BuildScopeIdIndexMap(scopeInfos_);

  // Print all scopes detail
  PrintAllScopesDetail(scopeInfos_);

  // Analysis section
  SK_LOGI("Scope Break Reason Analysis");
  for (size_t i = 0; i < scopeInfos_.size(); ++i) {
    const auto &scope = scopeInfos_[i];
    const auto &breakInfo = scope.GetBreakInfo();
    const auto &extInfo = scope.GetExtInfo();

    PrintScopeSummary(scope, i);
    SK_LOGI("  BreakInfo: %s", breakInfo.Format().c_str());

    if (breakInfo.GetParentScopeId() != INVALID_SCOPE_ID) {
      ScopeBreakInfo rootInfo = FindRootBreakInfo(scope, scopeIdToIdx, scopeInfos_);
      SK_LOGI("  Root BreakInfo: %s", rootInfo.Format().c_str());
    }

    PrintTriggerNode(breakInfo, graph_);
    PrintKernelNodes(ScopeSplitPass::GetKernelNodeIds(scope));
  }

  // Summary statistics
  auto rootReasonCounts = CountRootBreakReasons(scopeInfos_, scopeIdToIdx);
  PrintRootBreakReasonSummary(rootReasonCounts);

  SK_LOGI("End of Scope Break Reason Report");
}
