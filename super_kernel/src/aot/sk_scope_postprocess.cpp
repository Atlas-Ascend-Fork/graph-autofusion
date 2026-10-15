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
 * \file sk_scope_postprocess.cpp
 * \brief Post-processing functions for SK scope
 */

#include "acl/acl.h"
#include "securec.h"
#include "sk_resource_manager.h"
#include "sk_scope_postprocess.h"

namespace {

SuperKernelBaseNode *AdvanceNodeWithinScope(SuperKernelGraph &graph, SuperKernelBaseNode *node, uint64_t tailNodeId,
                                            uint32_t stepCount) {
  if (node == nullptr) {
    SK_LOGI("advance node skipped: start node is nullptr, tailNodeId=%lu, stepCount=%u", tailNodeId, stepCount);
    return nullptr;
  }

  SK_LOGI("advance node begin: startNodeId=%lu, tailNodeId=%lu, stepCount=%u", node->GetNodeId(), tailNodeId,
          stepCount);
  while (node != nullptr && stepCount > 0) {
    if (node->GetNodeId() == tailNodeId || node->GetNextNodeId() == INVALID_TASK_ID) {
      SK_LOGI("advance node stopped at boundary: currentNodeId=%lu, tailNodeId=%lu, remainStep=%u", node->GetNodeId(),
              tailNodeId, stepCount);
      return nullptr;
    }
    node = graph.GetNodeById(node->GetNextNodeId());
    if (node == nullptr) {
      SK_LOGI("advance node unsuccessful: next node not found in graph");
      return nullptr;
    }
    --stepCount;
  }
  SK_LOGI("advance node end: resultNodeId=%lu, remainStep=%u", node->GetNodeId(), stepCount);
  return node;
}

uint64_t FindKernelNodeWithFrontReserve(SuperKernelGraph &graph, SuperKernelBaseNode *headNode, uint64_t tailNodeId,
                                        uint32_t frontReserveCount) {
  // Search range is inclusive: after skipping reserved prefix nodes, the
  // remaining candidate window is [curNode, tailNodeId]. Therefore curNode
  // may legally be the tail node when the stream has only one remaining node.
  SK_LOGI("find kernel with reserve begin: headNodeId=%lu, tailNodeId=%lu, frontReserveCount=%u",
          headNode == nullptr ? INVALID_TASK_ID : headNode->GetNodeId(), tailNodeId, frontReserveCount);
  auto *curNode = AdvanceNodeWithinScope(graph, headNode, tailNodeId, frontReserveCount);
  while (curNode != nullptr) {
    if (curNode->GetNodeType() == SkNodeType::NODE_KERNEL) {
      SK_LOGI("find kernel with reserve success: kernelNodeId=%lu", curNode->GetNodeId());
      return curNode->GetNodeId();
    }
    if (curNode->GetNodeId() == tailNodeId || curNode->GetNextNodeId() == INVALID_TASK_ID) {
      SK_LOGI("find kernel with reserve stopped at boundary: currentNodeId=%lu, tailNodeId=%lu, nextNodeId=%lu",
              curNode->GetNodeId(), tailNodeId, curNode->GetNextNodeId());
      break;
    }
    curNode = graph.GetNodeById(curNode->GetNextNodeId());
  }
  SK_LOGI("find kernel with reserve unsuccessful: headNodeId=%lu, tailNodeId=%lu, frontReserveCount=%u",
          headNode == nullptr ? INVALID_TASK_ID : headNode->GetNodeId(), tailNodeId, frontReserveCount);
  return INVALID_TASK_ID;
}

bool HasAddrValue(const SuperKernelBaseNode &node) {
  switch (node.GetNodeType()) {
    case SkNodeType::NODE_NOTIFY:
    case SkNodeType::NODE_WAIT:
    case SkNodeType::NODE_RESET:
      return node.GetNodeInfos().syncInfos.addrValue != nullptr;
    default:
      return false;
  }
}

bool ShouldCancelNotify(const SuperKernelBaseNode &node) {
  // Regular notify nodes without pre-applied addrValue can follow the default cancellation rule.
  if (!HasAddrValue(node)) {
    return true;
  }
  // For notify nodes with pre-applied addrValue, keep write-only VALUE_WRITE notify.
  // A non-empty correspondingWaitNodeIds means this notify originally had paired waits.
  // Reaching eventCounts[eventId] == 0 means those waits are cancelled in this scope,
  // so the paired notify can be cancelled as well.
  return !node.GetCorrespondingWaitNodeIds().empty();
}

std::vector<SuperKernelBaseNode *> FilterCancelledNodes(const std::vector<SuperKernelBaseNode *> &nodes) {
  // Constraints:
  // 1) Within one scope, each event has at most one NOTIFY and may have multiple WAIT nodes.
  // 2) All WAIT nodes for that event are recorded in syncInfos.correspondingWaitNodeIds.

  std::vector<SuperKernelBaseNode *> filteredNodes;
  filteredNodes.reserve(nodes.size());
  // Core invariant per eventId after step 1:
  // eventCounts[eventId] = expected_wait_count_from_notify - observed_wait_count.
  // A value of 0 means notify-side expectation matches observed waits exactly.
  std::unordered_map<uint64_t, int64_t> eventCounts;

  // step 1: accumulate expected-vs-observed WAIT balance per eventId.
  for (size_t i = 0; i < nodes.size(); i++) {
    auto &curNode = nodes[i];
    if (curNode->GetNodeType() == SkNodeType::NODE_NOTIFY) {
      eventCounts[curNode->GetEventId()] += curNode->GetNodeInfos().syncInfos.correspondingWaitNodeIds.size();
    } else if (curNode->GetNodeType() == SkNodeType::NODE_WAIT) {
      eventCounts[curNode->GetEventId()]--;
    }
  }

  // step 2:
  // WAIT cancellation condition: corresponding NOTIFY exists.
  // NOTIFY cancellation condition: all WAITs of this event are cancellable.
  for (size_t i = 0; i < nodes.size(); i++) {
    auto &curNode = nodes[i];
    auto eventId = curNode->GetEventId();
    auto nodeType = curNode->GetNodeType();
    // Notify and Reset is removable only when expected and observed waits are fully matched.
    if (nodeType == SkNodeType::NODE_NOTIFY && eventCounts[eventId] == 0) {
      if (ShouldCancelNotify(*curNode)) {
        SK_LOGI("Event[0x%lx] cancelled in scope processing: NOTIFY node info : %s", eventId,
                curNode->Format().c_str());
        continue;
      }
    } else if (nodeType == SkNodeType::NODE_RESET && eventCounts.count(eventId) && eventCounts[eventId] == 0) {
      SK_LOGI("Event[0x%lx] cancelled in scope processing: RESET node info : %s", eventId, curNode->Format().c_str());
      continue;
      // WAIT is removable while the per-event balance remains non-negative.
      // Under the constraints above, a negative value means this scope has WAITs
      // for this event but no corresponding NOTIFY.
    } else if (nodeType == SkNodeType::NODE_WAIT && eventCounts[eventId] >= 0) {
      SK_LOGI("Event[0x%lx] cancelled in scope processing: WAIT node info : %s", eventId, curNode->Format().c_str());
      continue;
    }
    if (!curNode->IsScopeNode()) {
      filteredNodes.push_back(curNode);
    }
  }
  SK_LOGI("scope processing filtered nodes: %zu -> %zu (%zu cancelled)", nodes.size(), filteredNodes.size(),
          nodes.size() - filteredNodes.size());
  return filteredNodes;
}

bool EnsureStreamCapacity(const ScopeStreamInfo &streamInfo, const std::vector<aclmdlRITaskParams> &customParams) {
  SK_LOGI("stream capacity check: streamId=%u, nodeSize=%lu, customParamSize=%zu", streamInfo.streamIdx,
          streamInfo.nodeSize, customParams.size());
  if (customParams.size() > streamInfo.nodeSize) {
    SK_LOGE("scope processing overflow: streamId=%u, nodeSize=%lu, customParamSize=%zu", streamInfo.streamIdx,
            streamInfo.nodeSize, customParams.size());
    return false;
  }
  return true;
}

bool ProcessFrontWaitForStream(SuperKernelGraph &graph, ScopeExtInfo &extInfo,
                               const std::vector<ScopeStreamInfo> &scopeStreamInfos, std::vector<StreamPostPlan> &plans,
                               uint32_t curStreamIdx, uint64_t lastNodeId, uint32_t &needFrontWaitCount,
                               uint32_t &prevWaitStreamIdx) {
  auto &scopeStreamInfo = scopeStreamInfos[curStreamIdx];
  SK_LOGI("front-wait process begin: streamId=%u, prevWaitStreamIdx=%u, remainFrontWait=%u", scopeStreamInfo.streamIdx,
          prevWaitStreamIdx, needFrontWaitCount);
  needFrontWaitCount--;
  SK_LOGI("front-wait process state updated: streamId=%u, remainFrontWaitAfterDec=%u", scopeStreamInfo.streamIdx,
          needFrontWaitCount);

  // apply value memory addr
  void *addr = nullptr;
  aclError allocRet = SkResourceManager::ValueMemory(&addr);
  if (allocRet != ACL_SUCCESS || addr == nullptr) {
    SK_LOGE("front-wait value memory alloc failed: streamId=%u, ret=%d", scopeStreamInfo.streamIdx, allocRet);
    return false;
  }

  // create resetNode for sk optimize
  auto resetNode =
      SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(nullptr), ACL_MODEL_RI_TASK_EVENT_RESET,
                                         INVALID_TASK_ID, scopeStreamInfo.streamIdx, INVALID_STREAM_ID, lastNodeId);
  resetNode->SetNodeType(SkNodeType::NODE_RESET);
  resetNode->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
  resetNode->nodeInfos.syncInfos.addrValue = addr;
  // record resetNode for sk optimize
  extInfo.eventNodes.emplace_back(std::move(resetNode));

  // cur stream add record event task
  aclmdlRITaskParams notifyParams = {};
  notifyParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  notifyParams.valueWriteTaskParams.devAddr = addr;
  notifyParams.valueWriteTaskParams.value = SK_DEFAULT_NOTIFY_VALUE;
  extInfo.customParamsList[curStreamIdx].emplace_back(notifyParams);

  // prev stream add wait event task
  aclmdlRITaskParams waitParams = {};
  waitParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  waitParams.valueWaitTaskParams.devAddr = addr;
  waitParams.valueWaitTaskParams.value = SK_DEFAULT_WAIT_VALUE;
  waitParams.valueWaitTaskParams.flag = static_cast<uint32_t>(SkMemoryWaitFlag::EQ);
  extInfo.customParamsList[prevWaitStreamIdx].emplace(extInfo.customParamsList[prevWaitStreamIdx].begin(), waitParams);
  if (!EnsureStreamCapacity(scopeStreamInfos[curStreamIdx], extInfo.customParamsList[curStreamIdx])) {
    SK_LOGE("front-wait capacity check failed: curStreamId=%u, nodeSize=%lu, customParamSize=%zu",
            scopeStreamInfo.streamIdx, scopeStreamInfo.nodeSize, extInfo.customParamsList[curStreamIdx].size());
    return false;
  }
  if (!EnsureStreamCapacity(scopeStreamInfos[prevWaitStreamIdx], extInfo.customParamsList[prevWaitStreamIdx])) {
    SK_LOGE("front-wait capacity check failed: prevStreamId=%u, nodeSize=%lu, customParamSize=%zu",
            scopeStreamInfos[prevWaitStreamIdx].streamIdx, scopeStreamInfos[prevWaitStreamIdx].nodeSize,
            extInfo.customParamsList[prevWaitStreamIdx].size());
    return false;
  }

  // update info for next step
  prevWaitStreamIdx = curStreamIdx;
  return true;
}

bool ProcessBackBlockForStream(ScopeExtInfo &extInfo, const std::vector<ScopeStreamInfo> &scopeStreamInfos,
                               std::vector<StreamPostPlan> &plans, uint32_t curStreamIdx, uint64_t lastNodeId) {
  auto &scopeStreamInfo = scopeStreamInfos[curStreamIdx];
  SK_LOGI("back-block process begin: streamId=%u", scopeStreamInfo.streamIdx);

  // apply value memory addr
  void *addr = nullptr;
  aclError allocRet = SkResourceManager::ValueMemory(&addr);
  if (allocRet != ACL_SUCCESS || addr == nullptr) {
    SK_LOGE("back-block value memory alloc failed: streamId=%u, ret=%d", scopeStreamInfo.streamIdx, allocRet);
    return false;
  }

  // create notifyNode for sk optimize
  auto notifyNode =
      SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(nullptr), ACL_MODEL_RI_TASK_EVENT_RECORD,
                                         INVALID_TASK_ID, scopeStreamInfo.streamIdx, INVALID_STREAM_ID, lastNodeId);
  notifyNode->SetNodeType(SkNodeType::NODE_NOTIFY);
  notifyNode->nodeInfos.syncInfos.eventId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(addr));
  notifyNode->nodeInfos.syncInfos.addrValue = addr;

  // record notifyNode for sk optimize
  extInfo.eventNodes.emplace_back(std::move(notifyNode));

  // cur stream add wait event task
  aclmdlRITaskParams waitParams = {};
  waitParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
  waitParams.valueWaitTaskParams.devAddr = addr;
  waitParams.valueWaitTaskParams.value = SK_DEFAULT_WAIT_VALUE;
  waitParams.valueWaitTaskParams.flag = static_cast<uint32_t>(SkMemoryWaitFlag::EQ);
  extInfo.customParamsList[curStreamIdx].emplace_back(waitParams);

  // cur stream add reset event task
  aclmdlRITaskParams resetParams = {};
  resetParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
  resetParams.valueWriteTaskParams.devAddr = addr;
  resetParams.valueWriteTaskParams.value = SK_DEFAULT_RESET_VALUE;
  extInfo.customParamsList[curStreamIdx].emplace_back(resetParams);
  if (!EnsureStreamCapacity(scopeStreamInfo, extInfo.customParamsList[curStreamIdx])) {
    SK_LOGE("back-block capacity check failed: streamId=%u, nodeSize=%lu, customParamSize=%zu",
            scopeStreamInfo.streamIdx, scopeStreamInfo.nodeSize, extInfo.customParamsList[curStreamIdx].size());
    return false;
  }
  return true;
}

struct StreamCandidates {
  std::vector<uint32_t> mainStream;
  std::vector<uint32_t> subStream;
  std::vector<uint32_t> subStreamEntry;
};

bool CollectStreamCandidates(SuperKernelGraph &graph, const std::vector<ScopeStreamInfo> &scopeStreamInfos,
                             std::vector<StreamPostPlan> &plans, uint32_t needFrontWaitCount,
                             StreamCandidates &candidates) {
  const uint32_t streamCount = scopeStreamInfos.size();
  for (uint32_t curStreamIdx = 0; curStreamIdx < streamCount; ++curStreamIdx) {
    bool isMainStreamCandidate = false;
    bool isSubStreamCandidate = false;
    bool isSubStreamEntryCandidate = false;

    // Cache current stream info to avoid repeated indexing
    const auto &curStreamInfo = scopeStreamInfos[curStreamIdx];
    const auto &curPlan = plans[curStreamIdx];

    // Calculate front wait count for other streams (prevent unsigned underflow)
    const uint32_t curFrontWaitCount = curPlan.needFrontWait ? 1U : 0U;
    const uint32_t otherFrontWaitCount =
        (needFrontWaitCount >= curFrontWaitCount) ? (needFrontWaitCount - curFrontWaitCount) : 0U;
    const uint32_t mainFrontReserveCount = otherFrontWaitCount > 0 ? 1U : 0U;

    uint64_t candidateNodeId = FindKernelNodeWithFrontReserve(graph, graph.GetNodeById(curStreamInfo.headNodeIdx),
                                                              curStreamInfo.tailNodeIdx, mainFrontReserveCount);
    if (candidateNodeId != INVALID_TASK_ID) {
      candidates.mainStream.push_back(curStreamIdx);
      plans[curStreamIdx].candidateNodeId = candidateNodeId;
      isMainStreamCandidate = true;
      SK_LOGI("main stream candidate added: streamId=%u, candidateNodeId=%lu", curStreamInfo.streamIdx,
              candidateNodeId);
    }

    // Each frontWait/backBlock requires 2 nodes: frontWait for notify/wait, backBlock for wait/reset
    const uint32_t frontWaitNodeCount = 2U * (curPlan.needFrontWait ? 1U : 0U);
    const uint32_t backBlockNodeCount = 2U * (curPlan.needBackBlock ? 1U : 0U);
    const uint32_t subStreamNeedCount = frontWaitNodeCount + backBlockNodeCount;

    if (subStreamNeedCount <= curStreamInfo.nodeSize) {
      candidates.subStream.push_back(curStreamIdx);
      isSubStreamCandidate = true;
      SK_LOGI("sub stream candidate added: streamId=%u, needNodeCount=%u", curStreamInfo.streamIdx, subStreamNeedCount);
    }

    // SubStreamEntry requires one less node (can reuse frontWait from entry stream)
    const uint32_t subStreamEntryNeedCount = subStreamNeedCount - curFrontWaitCount;
    if (subStreamEntryNeedCount <= curStreamInfo.nodeSize) {
      candidates.subStreamEntry.push_back(curStreamIdx);
      isSubStreamEntryCandidate = true;
      SK_LOGI("sub stream entry candidate added: streamId=%u, needNodeCount=%u", curStreamInfo.streamIdx,
              subStreamEntryNeedCount);
    }

    if (!isMainStreamCandidate && !isSubStreamCandidate && !isSubStreamEntryCandidate) {
      SK_LOGI("streamId=%u does not meet the requirements for any candidate type:", curStreamInfo.streamIdx);
      SK_LOGI("1. mainStream: no candidate node found (reserveRequirement=%u)", mainFrontReserveCount);
      SK_LOGI("2. subStream: nodeSize=%lu, required=%u", curStreamInfo.nodeSize, subStreamNeedCount);
      SK_LOGI("3. subStreamEntry: nodeSize=%lu, required=%u", curStreamInfo.nodeSize, subStreamEntryNeedCount);
      return false;
    }
  }
  return true;
}

bool SelectMainAndEntryStream(const std::vector<StreamPostPlan> &plans, const StreamCandidates &candidates,
                              uint32_t streamCount, uint32_t &mainStreamIdx, uint32_t &entrySubStreamIdx,
                              uint64_t &skMainNodeId) {
  for (auto candidateMainStreamIdx : candidates.mainStream) {
    for (auto candidateEntrySubStreamIdx : candidates.subStreamEntry) {
      if (candidateEntrySubStreamIdx == candidateMainStreamIdx) {
        continue;
      }
      size_t localStreamCnt = candidates.subStream.size();
      auto itMain = find(candidates.subStream.begin(), candidates.subStream.end(), candidateMainStreamIdx);
      if (itMain == candidates.subStream.end()) {
        ++localStreamCnt;
      }
      auto itEntry = find(candidates.subStream.begin(), candidates.subStream.end(), candidateEntrySubStreamIdx);
      if (itEntry == candidates.subStream.end()) {
        ++localStreamCnt;
      }
      if (localStreamCnt == streamCount) {
        mainStreamIdx = candidateMainStreamIdx;
        skMainNodeId = plans[candidateMainStreamIdx].candidateNodeId;
        entrySubStreamIdx = candidateEntrySubStreamIdx;
        return true;
      }
    }
  }
  return false;
}

void BuildSubStreamOrder(const StreamCandidates &candidates, uint32_t mainStreamIdx, uint32_t entrySubStreamIdx,
                         std::vector<uint32_t> &subStreamOrder) {
  subStreamOrder.clear();
  for (auto curStreamIdx : candidates.subStream) {
    if (curStreamIdx != mainStreamIdx && curStreamIdx != entrySubStreamIdx) {
      subStreamOrder.push_back(curStreamIdx);
    }
  }
  subStreamOrder.push_back(entrySubStreamIdx);
}

bool GetMainAndSubStreamOrder(SuperKernelGraph &graph, std::vector<StreamPostPlan> &plans,
                              const std::vector<ScopeStreamInfo> &scopeStreamInfos, ScopeExtInfo &extInfo,
                              uint32_t needFrontWaitCount, uint32_t &mainStreamIdx,
                              std::vector<uint32_t> &subStreamOrder) {
  const uint32_t streamCount = scopeStreamInfos.size();
  StreamCandidates candidates;
  subStreamOrder.clear();
  subStreamOrder.reserve(streamCount);

  if (!CollectStreamCandidates(graph, scopeStreamInfos, plans, needFrontWaitCount, candidates)) {
    extInfo.processStatus = ScopeProcessStatus::RESOURCE_INSUFFICIENT;
    return false;
  }

  // Handle single stream case
  if (streamCount == 1) {
    if (plans[0].candidateNodeId != INVALID_TASK_ID) {
      extInfo.skMainNodeId = plans[0].candidateNodeId;
      return true;
    }
    SK_LOGE("single stream in scope but no candidate main node found, streamId=%u", scopeStreamInfos[0].streamIdx);
    extInfo.processStatus = ScopeProcessStatus::NO_TARGET_NODE;
    return false;
  }

  // Select main and entry stream
  uint32_t entrySubStreamIdx = INVALID_STREAM_ID;
  if (!SelectMainAndEntryStream(plans, candidates, streamCount, mainStreamIdx, entrySubStreamIdx,
                                extInfo.skMainNodeId)) {
    SK_LOGI("unable to select main and entry streams due to insufficient stream task slots, skip update");
    extInfo.processStatus = ScopeProcessStatus::RESOURCE_INSUFFICIENT;
    return false;
  }
  SK_LOGI("main stream and entry sub stream selected: mainStreamIdx=%u, entrySubStreamIdx=%u", mainStreamIdx,
          entrySubStreamIdx);

  BuildSubStreamOrder(candidates, mainStreamIdx, entrySubStreamIdx, subStreamOrder);
  return true;
}

bool ValidateEventInfo(SuperKernelGraph &graph, SuperKernelBaseNode *eventNode, const EventInfos *&eventInfos) {
  auto eventId = eventNode->GetEventId();
  eventInfos = graph.GetEventInfo(eventId);
  if (eventInfos == nullptr) {
    SK_LOGE("event not found in graph: event infos=%s", eventNode->Format().c_str());
    return false;
  }
  if (eventInfos->notifyNodeId == INVALID_TASK_ID || eventInfos->waitNodeIdList.empty() ||
      eventInfos->resetNodeIdList.empty()) {
    SK_LOGE("event syncInfos invalid: eventId=0x%lx, NotifyNodeId=%lu, WaitNodeIdsSize=%zu, ResetNodeIdsSize=%zu",
            eventNode->GetEventId(), eventInfos->notifyNodeId, eventInfos->waitNodeIdList.size(),
            eventInfos->resetNodeIdList.size());
    return false;
  }
  return true;
}

bool AllocateEventMemory(void *&addr, uint64_t eventId) {
  if (addr != nullptr) {
    SK_LOGI("event memory already applied: eventId=0x%lx, addr=%p", eventId, addr);
    return true;
  }
  SK_LOGI("event memory allocated start ...");
  aclError allocRet = SkResourceManager::ValueMemory(&addr);
  if (allocRet != ACL_SUCCESS || addr == nullptr) {
    SK_LOGE("event memory alloc failed: eventId=0x%lx, ret=%d", eventId, allocRet);
    return false;
  }
  return true;
}

bool UpdateEventNodeAddr(SuperKernelGraph &graph, uint64_t nodeId, void *addr,
                         std::vector<SuperKernelBaseNode *> &needUpdateNodes, const char *nodeType) {
  auto node = graph.GetNodeById(nodeId);
  if (node == nullptr) {
    SK_LOGE("%s event node not found in graph during event memory apply: nodeId=%lu", nodeType, nodeId);
    return false;
  }
  node->nodeInfos.syncInfos.addrValue = addr;
  needUpdateNodes.emplace_back(node);
  SK_LOGI("Updated %s node addrValue: nodeId=%lu, addr=%p", nodeType, node->GetNodeId(), addr);
  return true;
}

bool ApplyEventMemoryResource(SuperKernelGraph &graph, SuperKernelBaseNode *eventNode,
                              std::vector<SuperKernelBaseNode *> &needUpdateNodes) {
  auto &syncInfos = eventNode->nodeInfos.syncInfos;
  const EventInfos *eventInfos = nullptr;
  if (!ValidateEventInfo(graph, eventNode, eventInfos)) {
    return false;
  }

  void *addr = syncInfos.addrValue;
  if (!AllocateEventMemory(addr, syncInfos.eventId)) {
    return false;
  }

  // Update notify node
  if (!UpdateEventNodeAddr(graph, eventInfos->notifyNodeId, addr, needUpdateNodes, "notify")) {
    return false;
  }

  // Update wait nodes
  for (auto waitNodeId : eventInfos->waitNodeIdList) {
    if (!UpdateEventNodeAddr(graph, waitNodeId, addr, needUpdateNodes, "wait")) {
      return false;
    }
  }

  // Update reset nodes
  for (auto resetNodeId : eventInfos->resetNodeIdList) {
    if (!UpdateEventNodeAddr(graph, resetNodeId, addr, needUpdateNodes, "reset")) {
      return false;
    }
  }

  SK_LOGI("event memory allocated end: eventId=0x%lx, addr=%p", syncInfos.eventId, addr);
  return addr != nullptr;
}

uint32_t GetKernelNodeCount(const std::vector<SuperKernelBaseNode *> &nodes) {
  uint32_t kernelCnt = 0;
  for (auto &node : nodes) {
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      kernelCnt++;
    }
  }
  return kernelCnt;
}

}  // namespace

bool SuperKernelScopePostProcessor::ValidateScopeStreamNodes(const SuperKernelScopeInfo &scopeInfo) {
  for (const auto &streamInfo : scopeInfo.GetScopeStreamInfos()) {
    uint64_t curNodeId = streamInfo.headNodeIdx;
    while (curNodeId != INVALID_TASK_ID) {
      auto *curNode = graph.GetNodeById(curNodeId);
      if (curNode == nullptr) {
        SK_LOGE("node not found during scope post process: nodeId=%lu, streamId=%u", curNodeId, streamInfo.streamIdx);
        return false;
      }
      if (!curNode->IsFusible()) {
        SK_LOGE("node is not fusible during scope post process: nodeId=%lu, streamId=%u", curNodeId,
                streamInfo.streamIdx);
        return false;
      }
      if (curNodeId == streamInfo.tailNodeIdx) {
        break;
      }
      curNodeId = curNode->GetNextNodeId();
    }
  }
  return true;
}

bool SuperKernelScopePostProcessor::ApplyEventMemoryForFilteredNodes(
    std::vector<SuperKernelBaseNode *> &filteredNodes, std::vector<SuperKernelBaseNode *> &needUpdateNodes) {
  for (auto node : filteredNodes) {
    auto nodeType = node->GetNodeType();
    if (nodeType == SkNodeType::NODE_NOTIFY || nodeType == SkNodeType::NODE_WAIT ||
        nodeType == SkNodeType::NODE_RESET) {
      if (HasAddrValue(*node)) {
        SK_LOGI("event memory resource already exists, skip: nodeId=%lu, nodeType=%d", node->GetNodeId(),
                static_cast<int>(nodeType));
        continue;
      }
      if (!ApplyEventMemoryResource(graph, node, needUpdateNodes)) {
        SK_LOGE("event memory resource apply failed during scope processing: nodeId=%lu, nodeType=%d",
                node->GetNodeId(), static_cast<int>(nodeType));
        return false;
      }
    }
  }
  SK_LOGI("applied event memory resource for filtered nodes, needUpdateNodesSize=%zu", needUpdateNodes.size());
  return true;
}

bool SuperKernelScopePostProcessor::CollectStreamBoundaryPlans(const SuperKernelScopeInfo &scopeInfo,
                                                               std::vector<StreamPostPlan> &plans,
                                                               uint32_t &needFrontWaitCount) {
  uint32_t streamCount = scopeInfo.GetScopeStreamInfos().size();
  for (uint32_t curStreamIdx = 0; curStreamIdx < streamCount; ++curStreamIdx) {
    SK_LOGI("collect stream info: streamIdx=%u, nodeSize=%lu", curStreamIdx,
            scopeInfo.GetScopeStreamInfos()[curStreamIdx].nodeSize);
    const auto &scopeStreamInfo = scopeInfo.GetScopeStreamInfos()[curStreamIdx];

    auto *headNode = graph.GetNodeById(scopeStreamInfo.headNodeIdx);
    auto *tailNode = graph.GetNodeById(scopeStreamInfo.tailNodeIdx);
    if (headNode == nullptr) {
      SK_LOGE("head node not found in graph during scope processing: head=%lu", scopeStreamInfo.headNodeIdx);
      return false;
    }
    if (tailNode == nullptr) {
      SK_LOGE("tail node not found in graph during scope processing: tail=%lu", scopeStreamInfo.tailNodeIdx);
      return false;
    }

    if (headNode->GetPreNodeId() != INVALID_TASK_ID) {
      plans[curStreamIdx].needFrontWait = true;
      needFrontWaitCount++;
    }
    if (tailNode->GetNextNodeId() != INVALID_TASK_ID) {
      plans[curStreamIdx].needBackBlock = true;
    }
    SK_LOGI(
        "stream plan collected: streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, needFrontWait=%d, "
        "needBackBlock=%d",
        scopeStreamInfo.streamIdx, scopeStreamInfo.headNodeIdx, scopeStreamInfo.tailNodeIdx, scopeStreamInfo.nodeSize,
        plans[curStreamIdx].needFrontWait, plans[curStreamIdx].needBackBlock);
  }
  SK_LOGI("collect stream done: streamCount=%u, needFrontWaitCount=%u", streamCount, needFrontWaitCount);
  return true;
}

bool SuperKernelScopePostProcessor::ProcessSubStreamSyncEvents(
    SuperKernelScopeInfo &scopeInfo, ScopeExtInfo &tempExtInfo, std::vector<StreamPostPlan> &plans,
    uint32_t mainStreamIdx, const std::vector<uint32_t> &subStreamOrder, uint32_t needFrontWaitCount) {
  uint32_t prevWaitStreamIdx = mainStreamIdx;
  uint64_t lastNodeId = tempExtInfo.filteredNodes.back()->GetNodeId();
  SK_LOGI("scope processing front-wait and back-block begin: lastNodeId=%lu", lastNodeId);

  auto &scopeStreamInfos = scopeInfo.GetScopeStreamInfos();
  for (uint32_t curStreamIdx : subStreamOrder) {
    if (plans[curStreamIdx].needFrontWait &&
        !ProcessFrontWaitForStream(graph, tempExtInfo, scopeStreamInfos, plans, curStreamIdx, lastNodeId,
                                   needFrontWaitCount, prevWaitStreamIdx)) {
      SK_LOGE("process front-wait failed streamId=%u, nodeSize=%lu, FrontWait=%u",
              scopeStreamInfos[curStreamIdx].streamIdx, scopeStreamInfos[curStreamIdx].nodeSize,
              plans[curStreamIdx].needFrontWait);
      tempExtInfo.processStatus = ScopeProcessStatus::RESOURCE_INSUFFICIENT;
      return false;
    }

    if (plans[curStreamIdx].needBackBlock &&
        !ProcessBackBlockForStream(tempExtInfo, scopeStreamInfos, plans, curStreamIdx, lastNodeId)) {
      SK_LOGE("process back-block failed streamId=%u, nodeSize=%lu, BackBlock=%u",
              scopeStreamInfos[curStreamIdx].streamIdx, scopeStreamInfos[curStreamIdx].nodeSize,
              plans[curStreamIdx].needBackBlock);
      tempExtInfo.processStatus = ScopeProcessStatus::RESOURCE_INSUFFICIENT;
      return false;
    }
  }
  SK_LOGI("scope processing front-wait and back-block done for all sub streams");
  return true;
}

bool SuperKernelScopePostProcessor::FinalizePostProcess(SuperKernelScopeInfo &scopeInfo, ScopeExtInfo &tempExtInfo,
                                                        std::vector<SuperKernelBaseNode *> &needUpdateNodes) {
  graph.ExpandUpdateNodes(needUpdateNodes);
  SK_LOGI("expand update node done, needUpdateNodesSize=%zu", needUpdateNodes.size());

  size_t totalCustomParamSize = 0;
  for (const auto &customParams : tempExtInfo.customParamsList) {
    totalCustomParamSize += customParams.size();
  }
  SK_LOGI("scope processing end: streamCount=%u, skMainNodeId=%lu, eventNodeCount=%zu, totalCustomParamSize=%zu",
          scopeInfo.GetScopeStreamInfos().size(), tempExtInfo.skMainNodeId, tempExtInfo.eventNodes.size(),
          totalCustomParamSize);

  scopeInfo.SetExtInfo(std::move(tempExtInfo));
  scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::SUCCESS;
  return true;
}

bool SuperKernelScopePostProcessor::PostProcess(SuperKernelScopeInfo &scopeInfo) {
  scopeInfo.SetExtInfo(ScopeExtInfo{});
  // Step 1: Validate all nodes in scope streams
  if (!ValidateScopeStreamNodes(scopeInfo)) {
    scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::UNRECOVERABLE_FAIL;
    return false;
  }

  uint32_t streamCount = scopeInfo.GetScopeStreamInfos().size();
  SK_LOGI("scope processing begin: streamCount=%u, nodeCount=%zu", streamCount, scopeInfo.GetNodes().size());

  // Step 2: Filter cancelled nodes
  std::vector<SuperKernelBaseNode *> filteredNodes = FilterCancelledNodes(scopeInfo.GetNodes());
  if (filteredNodes.empty()) {
    SK_LOGI("scope unprocessable: no node remains after cancelling notify/wait pairs");
    scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::NO_TARGET_NODE;
    return false;
  }

  // Step 3: Skip scope when no kernel remains after filtering
  if (GetKernelNodeCount(filteredNodes) == 0) {
    SK_LOGI("scope unprocessable: no kernel node remains after filtering");
    scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::NO_TARGET_NODE;
    return false;
  }

  // Step 4: Initialize temp extInfo
  ScopeExtInfo tempExtInfo;
  tempExtInfo.filteredNodes = std::move(filteredNodes);
  tempExtInfo.skMainNodeId = INVALID_TASK_ID;
  tempExtInfo.customParamsList.resize(streamCount);

  std::vector<StreamPostPlan> plans(streamCount);
  uint32_t needFrontWaitCount = 0;
  std::vector<SuperKernelBaseNode *> needUpdateNodes;

  // Step 5: Collect stream boundary plans
  if (!CollectStreamBoundaryPlans(scopeInfo, plans, needFrontWaitCount)) {
    scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::UNRECOVERABLE_FAIL;
    return false;
  }

  // Step 6: Select main stream and sub stream order
  uint32_t mainStreamIdx = INVALID_STREAM_ID;
  std::vector<uint32_t> subStreamOrder;
  if (!GetMainAndSubStreamOrder(graph, plans, scopeInfo.GetScopeStreamInfos(), tempExtInfo, needFrontWaitCount,
                                mainStreamIdx, subStreamOrder)) {
    SK_LOGI("scope unprocessable: unable to select main stream and sub stream");
    scopeInfo.MutableExtInfo().processStatus = tempExtInfo.processStatus;
    return false;
  }
  SK_LOGI("select main stream and sub stream done.");

  // Step 7: Process sync events for sub streams
  if (!ProcessSubStreamSyncEvents(scopeInfo, tempExtInfo, plans, mainStreamIdx, subStreamOrder, needFrontWaitCount)) {
    scopeInfo.MutableExtInfo().processStatus = tempExtInfo.processStatus;
    return false;
  }

  // Step 8: Apply event memory resource
  if (!ApplyEventMemoryForFilteredNodes(tempExtInfo.filteredNodes, needUpdateNodes)) {
    scopeInfo.MutableExtInfo().processStatus = ScopeProcessStatus::UNRECOVERABLE_FAIL;
    return false;
  }

  // Step 9: Finalize and assign result
  return FinalizePostProcess(scopeInfo, tempExtInfo, needUpdateNodes);
}
