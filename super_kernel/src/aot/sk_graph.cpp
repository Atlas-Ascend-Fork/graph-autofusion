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
 * \file sk_graph.cpp
 * \brief
 */

#include "sk_graph.h"
#include "sk_dump_json.h"
#include "sk_options_manager.h"
#include "sk_scope_split.h"
#include "super_kernel.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <bitset>
#include <unordered_map>

namespace {

uint64_t ResolveEventValue(const SuperKernelBaseNode *node) {
  const auto &syncInfos = node->GetNodeInfos().syncInfos;
  if (syncInfos.memoryValue != std::numeric_limits<uint64_t>::max()) {
    return syncInfos.memoryValue;
  }

  switch (node->GetNodeType()) {
    case SkNodeType::NODE_NOTIFY:
      return SK_DEFAULT_NOTIFY_VALUE;
    case SkNodeType::NODE_WAIT:
      return SK_DEFAULT_WAIT_VALUE;
    case SkNodeType::NODE_RESET:
      return SK_DEFAULT_RESET_VALUE;
    default:
      return 0;
  }
}

uint32_t ResolveEventWaitFlag(const SuperKernelBaseNode *node) {
  const auto &syncInfos = node->GetNodeInfos().syncInfos;
  if (node->GetNodeType() != SkNodeType::NODE_WAIT) {
    return SK_DEFAULT_WRITE_FLAG;
  }
  if (syncInfos.memoryWaitFlag != std::numeric_limits<uint32_t>::max()) {
    return syncInfos.memoryWaitFlag;
  }
  return static_cast<uint32_t>(SkMemoryWaitFlag::EQ);
}

bool IsNotifyByMemoryWaitRule(uint64_t writeMemoryValue, uint64_t memoryWaitValue, uint32_t waitFlag,
                              uint64_t eventId) {
  switch (static_cast<SkMemoryWaitFlag>(waitFlag)) {
    case SkMemoryWaitFlag::GEQ:
      return writeMemoryValue >= memoryWaitValue;
    case SkMemoryWaitFlag::EQ:
      return writeMemoryValue == memoryWaitValue;
    case SkMemoryWaitFlag::AND:
      return (writeMemoryValue & memoryWaitValue) != 0;
    case SkMemoryWaitFlag::NOR:
      return (~(writeMemoryValue | memoryWaitValue)) != 0;
    default:
      SK_LOGW("Unknown waitFlag %u for event 0x%lx, treated as reset node", waitFlag, eventId);
      return false;
  }
}

}  // namespace

std::string SuperKernelGraph::BitsetToString(const std::bitset<MAX_SCOPE_NUM> &bitset) const {
  std::string strTmp = bitset.to_string();
  if (scopeNameToIdx.size() == 0) {
    return "0";
  }
  std::string revStr(strTmp.rbegin(), strTmp.rend());
  return revStr.substr(0, scopeNameToIdx.size());
}

aclError SuperKernelGraph::Update() {
  if (!needUpdate && needUpdateNodes.empty()) {
    SK_LOGI("No update needed for SuperKernelGraph");
    return ACL_SUCCESS;
  }
  for (auto node : needUpdateNodes) {
    if (node == nullptr) {
      SK_LOGE("Null node found in needUpdateNodes");
      return ACL_ERROR_FAILURE;
    }
    if (!node->IsUpdated()) {
      aclmdlRITaskParams customParams = {};
      switch (node->GetNodeType()) {
        case SkNodeType::NODE_NOTIFY: {
          customParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
          customParams.valueWriteTaskParams.value = ResolveEventValue(node);
          customParams.valueWriteTaskParams.devAddr = node->nodeInfos.syncInfos.addrValue;
          break;
        }
        case SkNodeType::NODE_WAIT: {
          customParams.type = ACL_MODEL_RI_TASK_VALUE_WAIT;
          customParams.valueWaitTaskParams.value = ResolveEventValue(node);
          customParams.valueWaitTaskParams.flag = ResolveEventWaitFlag(node);
          customParams.valueWaitTaskParams.devAddr = node->nodeInfos.syncInfos.addrValue;
          break;
        }
        case SkNodeType::NODE_RESET: {
          customParams.type = ACL_MODEL_RI_TASK_VALUE_WRITE;
          customParams.valueWriteTaskParams.value = ResolveEventValue(node);
          customParams.valueWriteTaskParams.devAddr = node->nodeInfos.syncInfos.addrValue;
          break;
        }
        case SkNodeType::NODE_KERNEL:
          break;  // invalid node
        default:
          SK_LOGE("Unsupported node type for event update: %u", static_cast<uint32_t>(node->GetNodeType()));
          return ACL_ERROR_FAILURE;
      }
      UpdateContext ctx;
      ctx.customParams = &customParams;
      if (!node->Update(ctx)) {
        SK_LOGE("Failed to update event node %lu in stream %u", node->GetNodeId(), node->GetStreamIdxInGraph());
        return ACL_ERROR_FAILURE;
      }
    }
  }
  aclError ret = aclmdlRIUpdate(modelRI);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to update modelRI");
  }
  // Clear shape info memory after update completes
  ClearShapeInfoPtrList();
  return ret;
}

bool SuperKernelGraph::ExpandUpdateNodes(std::vector<SuperKernelBaseNode *> &customNodes) {
  for (auto *node : customNodes) {
    if (node != nullptr && needUpdateNodes.find(node) == needUpdateNodes.end()) {
      needUpdateNodes.insert(node);
      SK_LOGI("Insert into sk graph needUpdateNodes, node info: %s", node->Format().c_str());
    }
  }
  return true;
}

SuperKernelBaseNode *SuperKernelGraph::GetNodeById(uint64_t nodeId) const {
  auto it = graphMap.find(nodeId);
  if (it != graphMap.end()) {
    return it->second.get();
  }
  SK_LOGE("Node with id %lu not found in graph", nodeId);
  return nullptr;
}

uint64_t SuperKernelGraph::AllocateSyntheticNodeId() {
  if (maxNodeId_ >= INVALID_TASK_ID - 1U) {
    SK_LOGE("Failed to allocate synthetic node ID: node ID space exhausted, maxNodeId=%lu", maxNodeId_);
    return INVALID_TASK_ID;
  }
  return ++maxNodeId_;
}

bool SuperKernelGraph::AddNode(std::unique_ptr<SuperKernelBaseNode> node) {
  uint64_t nodeId = node->GetNodeId();
  uint64_t eventId = node->GetEventId();
  switch (node->GetNodeType()) {
    case SkNodeType::NODE_NOTIFY:
      if (!AddEventAssociateNotify(eventId, node.get())) {
        SK_LOGE("Failed to associate notify event 0x%lx with node %lu", eventId, nodeId);
        return false;
      }
      break;
    case SkNodeType::NODE_WAIT:
      if (!AddEventAssociateWait(eventId, node.get())) {
        SK_LOGE("Failed to associate wait event 0x%lx with node %lu", eventId, nodeId);
        return false;
      }
      break;
    case SkNodeType::NODE_RESET:
      if (!AddEventAssociateReset(eventId, node.get())) {
        SK_LOGE("Failed to associate reset event 0x%lx with node %lu", eventId, nodeId);
        return false;
      }
      break;
    case SkNodeType::NODE_MEMORY_WRITE:
      if (!AddMemoryAssociateWrite(eventId, node.get())) {
        SK_LOGE("Failed to associate memory write event 0x%lx with node %lu", eventId, nodeId);
        return false;
      }
      break;
    case SkNodeType::NODE_MEMORY_WAIT:
      if (!AddMemoryAssociateWait(eventId, node.get())) {
        SK_LOGE("Failed to associate memory wait event 0x%lx with node %lu", eventId, nodeId);
        return false;
      }
      break;
    default:
      break;
  }

  if (graphMap.find(nodeId) != graphMap.end()) {
    SuperKernelBaseNode *existingNode = graphMap[nodeId].get();
    SK_LOGE("Duplicate node ID detected! Node with id %lu already exists in graph", nodeId);
    SK_LOGE("  Existing node: %s", existingNode->Format().c_str());
    SK_LOGE("  New node to add: %s", node->Format().c_str());
    SK_LOGE("  Duplicate nodeId=%lu, Please check for duplicate node assignments in stream %lu", nodeId,
            node->GetStreamIdxInGraph());
    return false;
  }
  graphMap[nodeId] = std::move(node);
  maxNodeId_ = std::max(maxNodeId_, nodeId);
  return true;
}

bool SuperKernelGraph::AddMemoryAssociateWrite(uint64_t eventId, SuperKernelBaseNode *node) {
  auto &memoryInfo = memoryToNodes[eventId];
  uint64_t nodeId = node->GetNodeId();
  if (memoryInfo.writeNodeIdList.find(nodeId) != memoryInfo.waitNodeIdList.end()) {
    // Get the node information for duplicate memory write node binding
    SuperKernelBaseNode *existingNode = GetNodeById(nodeId);
    SK_LOGE("memory event 0x%lx already associated with this node, cannot reassociate!", eventId);
    SK_LOGE("  Duplicate memory write node: node_id=%lu, details=%s", nodeId,
            existingNode ? existingNode->Format().c_str() : "NOT_FOUND");
    SK_LOGE("  Please check for duplicate memory write node bindings in the graph.");
    return false;
  }
  memoryInfo.writeNodeIdList.insert(nodeId);
  return true;
}

bool SuperKernelGraph::AddMemoryAssociateWait(uint64_t eventId, SuperKernelBaseNode *node) {
  auto &memoryInfo = memoryToNodes[eventId];
  uint64_t nodeId = node->GetNodeId();
  if (memoryInfo.waitNodeIdList.find(nodeId) != memoryInfo.waitNodeIdList.end()) {
    // Get the node information for duplicate memory wait node binding
    SuperKernelBaseNode *existingNode = GetNodeById(nodeId);
    SK_LOGE("memory event 0x%lx already associated with this node, cannot reassociate!", eventId);
    SK_LOGE("  Duplicate memory wait node: node_id=%lu, details=%s", nodeId,
            existingNode ? existingNode->Format().c_str() : "NOT_FOUND");
    SK_LOGE("  Please check for duplicate memory wait node bindings in the graph.");
    return false;
  }
  memoryInfo.waitNodeIdList.insert(nodeId);
  return true;
}

bool SuperKernelGraph::AddEventAssociateNotify(uint64_t eventId, SuperKernelBaseNode *node) {
  auto &eventInfo = eventToNodes[eventId];
  if (eventInfo.notifyNodeId != INVALID_TASK_ID) {
    // Get the already bound node information
    SuperKernelBaseNode *existingNode = GetNodeById(eventInfo.notifyNodeId);
    SK_LOGE("Notify event 0x%lx already associated, cannot reassociate!", eventId);
    SK_LOGE("  Existing bound node: node_id=%lu, details=%s", eventInfo.notifyNodeId,
            existingNode ? existingNode->Format().c_str() : "NOT_FOUND");
    SK_LOGE("  Attempting to bind: node_id=%lu, details=%s", node->GetNodeId(), node->Format().c_str());
    SK_LOGE("  Please check for duplicate event bindings in the graph.");
    return false;
  }
  eventInfo.notifyNodeId = node->GetNodeId();
  return true;
}

bool SuperKernelGraph::AddEventAssociateWait(uint64_t eventId, SuperKernelBaseNode *node) {
  auto &eventInfo = eventToNodes[eventId];
  uint64_t nodeId = node->GetNodeId();
  if (eventInfo.waitNodeIdList.find(nodeId) != eventInfo.waitNodeIdList.end()) {
    // Get the node information for duplicate WAIT binding
    SuperKernelBaseNode *existingNode = GetNodeById(nodeId);

    SK_LOGE("Wait event 0x%lx already associated with this node, cannot reassociate!", eventId);
    SK_LOGE("  Duplicate WAIT node: node_id=%lu, details=%s", nodeId,
            existingNode ? existingNode->Format().c_str() : "NOT_FOUND");
    SK_LOGE("  Please check for duplicate WAIT event bindings in the graph.");
    return false;
  }
  eventInfo.waitNodeIdList.insert(nodeId);

  return true;
}

bool SuperKernelGraph::AddEventAssociateReset(uint64_t eventId, SuperKernelBaseNode *node) {
  auto &eventInfo = eventToNodes[eventId];
  uint64_t nodeId = node->GetNodeId();
  if (eventInfo.resetNodeIdList.find(nodeId) != eventInfo.resetNodeIdList.end()) {
    SK_LOGE("Reset event 0x%lx already associated, cannot reassociate!", eventId);
    SK_LOGE("  Existing bound node: node_id=%lu, details=%s", nodeId, node ? node->Format().c_str() : "NOT_FOUND");
    SK_LOGE("  Attempting to bind: node_id=%lu, details=%s", nodeId, node->Format().c_str());
    SK_LOGE("  Please check for duplicate RESET event bindings in the graph.");
    return false;
  }
  eventInfo.resetNodeIdList.insert(nodeId);
  return true;
}

/**
 * @brief Build event associations and establish send-receive relationships between nodes
 *
 * This function traverses all event information to establish direct send-receive relationships
 * between NOTIFY nodes and WAIT nodes associated with each event:
 * 1. For each event, find the nearest KERNEL node before the NOTIFY node as the sender node
 * 2. For each WAIT node, find the nearest KERNEL node after it as the receiver node
 * 3. Establish bidirectional association between sender and receiver nodes
 *
 * @return true Association establishment successful
 * @return false Association establishment failed (current implementation always returns true)
 *
 * @note This function is mainly used to build the task dependency graph within super kernel,
 *       facilitating subsequent graph optimization and execution
 */
bool SuperKernelGraph::AddEventAssociate() {
  SK_LOGI("Start to build event associations, total events: %zu", eventToNodes.size());

  uint32_t totalProcessed = 0;
  uint32_t skipDueToNoNotify = 0;
  uint32_t skipDueToNoSendNode = 0;
  uint32_t skipDueToNoWaitNode = 0;
  uint32_t successAssociations = 0;

  // Traverse all event information
  for (auto iter = eventToNodes.begin(); iter != eventToNodes.end(); iter++) {
    uint64_t eventId = iter->first;                                        // Event ID
    uint64_t notifyId = iter->second.notifyNodeId;                         // NOTIFY node ID
    std::unordered_set<uint64_t> waitIdSet = iter->second.waitNodeIdList;  // WAIT node ID set

    SK_LOGD("Processing event 0x%lx: notify=%lu, wait_count=%zu", eventId, notifyId, waitIdSet.size());

    // Skip if the event has no associated NOTIFY node
    if (INVALID_TASK_ID == notifyId) {
      SK_LOGD("Event 0x%lx has no notify node, skip", eventId);
      skipDueToNoNotify++;
      continue;
    }

    // Search backward from NOTIFY node to find the first KERNEL node as sender node
    // This ensures the sender is the actual compute kernel rather than event notification node
    auto nodeId = graphMap[notifyId]->GetPreNodeId();
    while (nodeId != INVALID_TASK_ID && graphMap[nodeId]->GetNodeType() != SkNodeType::NODE_KERNEL) {
      nodeId = graphMap[nodeId]->GetPreNodeId();
    }

    // Skip this event if no KERNEL node found before NOTIFY node
    if (nodeId == INVALID_TASK_ID) {
      SK_LOGW("Event 0x%lx: cannot find KERNEL node before notify node %lu, skip", eventId, notifyId);
      skipDueToNoSendNode++;
      continue;
    }

    auto sendNode = graphMap[nodeId].get();
    if (sendNode == nullptr) {
      SK_LOGE("Failed to cast node %lu to SuperKernelSendNode", nodeId);
      continue;
    }

    SK_LOGD("Event 0x%lx: send node found (id=%lu), starting to process %zu wait nodes", eventId, nodeId,
            waitIdSet.size());

    // Traverse all WAIT nodes associated with this event
    for (auto waitId : waitIdSet) {
      // Search forward from WAIT node to find the first KERNEL node as receiver node
      // This ensures the receiver is the actual compute kernel that needs synchronization
      while (waitId != INVALID_TASK_ID && graphMap[waitId]->GetNodeType() != SkNodeType::NODE_KERNEL) {
        waitId = graphMap[waitId]->GetNextNodeId();
      }

      // Skip this WAIT node if no KERNEL node found after it
      if (waitId == INVALID_TASK_ID) {
        SK_LOGW("Event 0x%lx: cannot find KERNEL node after wait node, skip one wait association", eventId);
        skipDueToNoWaitNode++;
        continue;
      }

      auto recvNode = graphMap[waitId].get();
      if (recvNode == nullptr) {
        SK_LOGE("Failed to cast node %lu to SuperKernelSendNode", waitId);
        continue;
      }

      // Establish bidirectional association:
      // 1. Receiver node records which sender nodes it needs to wait for
      // 2. Sender node records which receiver nodes it needs to notify
      recvNode->receiveNodeId.insert(nodeId);
      sendNode->sendToNodeId.insert(waitId);

      SK_LOGD("Event 0x%lx: built association - send(node %lu) -> wait(node %lu)", eventId, nodeId, waitId);
      successAssociations++;
    }

    totalProcessed++;
  }

  SK_LOGI(
      "Event association build completed: processed=%u, success=%u, "
      "skip(no_notify)=%u, skip(no_send)=%u, skip(no_wait)=%u",
      totalProcessed, successAssociations, skipDueToNoNotify, skipDueToNoSendNode, skipDueToNoWaitNode);

  return true;
}

void SuperKernelGraph::BuildEventNodeAssociations() {
  SK_LOGI("Starting to build wait node associations, total events: %zu", eventToNodes.size());
  for (const auto &it : eventToNodes) {
    const uint64_t eventId = it.first;
    const EventInfos &eventInfo = it.second;
    SK_LOGI("Processing event 0x%lx, notify nodeId: %lu, wait node size: %zu, reset node size: %zu", eventId,
            eventInfo.notifyNodeId, eventInfo.waitNodeIdList.size(), eventInfo.resetNodeIdList.size());
    if (eventInfo.notifyNodeId != INVALID_TASK_ID && !eventInfo.waitNodeIdList.empty()) {
      auto *notifyNode = GetNodeById(eventInfo.notifyNodeId);
      if (notifyNode != nullptr && (notifyNode->GetNodeType() == SkNodeType::NODE_NOTIFY)) {
        std::vector<uint64_t> waitNodeIds(eventInfo.waitNodeIdList.begin(), eventInfo.waitNodeIdList.end());
        notifyNode->SetCorrespondingWaitNodeIds(waitNodeIds);

        SK_LOGI("notify node %lu has %zu wait nodes", eventInfo.notifyNodeId, waitNodeIds.size());

        // Set corresponding notify node ID for each wait node
        for (uint64_t waitNodeId : waitNodeIds) {
          auto *waitNode = GetNodeById(waitNodeId);
          if (waitNode != nullptr && (waitNode->GetNodeType() == SkNodeType::NODE_WAIT)) {
            waitNode->SetCorrespondingNotifyNodeId(eventInfo.notifyNodeId);
            SK_LOGD("Associated wait node %lu with notify node %lu", waitNodeId, eventInfo.notifyNodeId);
          }
        }
      } else {
        SK_LOGE("Event 0x%lx: notify node %lu is invalid or not a notify node", eventId, eventInfo.notifyNodeId);
      }
    } else if (eventInfo.notifyNodeId != INVALID_TASK_ID && eventInfo.waitNodeIdList.empty()) {
      auto *notifyNode = GetNodeById(eventInfo.notifyNodeId);
      if (notifyNode != nullptr && notifyNode->GetNodeType() == SkNodeType::NODE_NOTIFY) {
        notifyNode->SetCorrespondingWaitNodeIds({});
        SK_LOGI("Event 0x%lx: notify node %lu has no wait node in modelRI, keep current fusible state", eventId,
                eventInfo.notifyNodeId);
      } else {
        SK_LOGE("Event 0x%lx: orphan notify node %lu is invalid or not a notify node", eventId, eventInfo.notifyNodeId);
      }
    }
    if (!eventInfo.resetNodeIdList.empty()) {
      for (auto resetNodeId : eventInfo.resetNodeIdList) {
        auto *resetNode = GetNodeById(resetNodeId);
        if (resetNode != nullptr && (resetNode->GetNodeType() == SkNodeType::NODE_RESET)) {
          resetNode->nodeInfos.syncInfos.correspondingResetNodeIds.assign(eventInfo.resetNodeIdList.begin(),
                                                                          eventInfo.resetNodeIdList.end());
          SK_LOGD("Associated reset node %lu has %lu reset nodes", resetNodeId, eventInfo.resetNodeIdList.size());
        }
      }
    }
  }
  SK_LOGI("Completed building all event node associations");
}

uint32_t SuperKernelGraph::GetValueBreakerBypass() const {
  if (opts_ == nullptr) {
    return ACLSK_VALUE_BREAKER_BYPASS_NONE;
  }
  const auto *option =
      static_cast<const AggressiveOptStrategiesOption *>(opts_->GetOption(aclskOptionType::AGGRESSIVE_OPT_STRATEGIES));
  return option == nullptr ? ACLSK_VALUE_BREAKER_BYPASS_NONE : option->GetValue().valueBreakerBypass;
}

bool SuperKernelGraph::ProcessMemoryWriteNodes(const uint64_t eventId, const MemoryInfos &memoryInfo,
                                               const uint64_t memoryWaitValue, const uint32_t waitFlag) {
  const uint32_t valueBreakerBypass = GetValueBreakerBypass();
  std::vector<uint64_t> notifyIdVec;
  std::vector<uint64_t> resetIdVec;

  // Iterate through all memory write nodes and classify them as notify/reset
  // according to the runtime-compatible memory wait rule.
  for (const auto writeNodeId : memoryInfo.writeNodeIdList) {
    auto *writeNode = GetNodeById(writeNodeId);
    // Only process valid MEMORY_WRITE type nodes
    if (writeNode == nullptr || writeNode->GetNodeType() != SkNodeType::NODE_MEMORY_WRITE) {
      continue;
    }
    const uint64_t writeMemoryValue = writeNode->nodeInfos.syncInfos.memoryValue;
    const bool isNotify = IsNotifyByMemoryWaitRule(writeMemoryValue, memoryWaitValue, waitFlag, eventId);

    // Classify nodes into corresponding vectors based on judgment result
    if (isNotify) {
      notifyIdVec.push_back(writeNodeId);
    } else {
      resetIdVec.push_back(writeNodeId);
    }
  }

  // value_breaker_bypass is a wait-oriented bitmask
  const bool enablePairedWaitBypass = (valueBreakerBypass & ACLSK_VALUE_BREAKER_BYPASS_PAIRED_WAIT) != 0;
  const bool enableUnpairedWaitBypass = (valueBreakerBypass & ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT) != 0;

  bool waitFusible = false;
  // check notify size and apply corresponding bypass policy
  if (notifyIdVec.size() > 1) {
    SK_LOGE("there exist multiple memory write nodes which are notify, it is illegal, eventId: 0x%lx", eventId);
    return false;
  } else if (notifyIdVec.size() == 1) {
    auto *writeNode = GetNodeById(notifyIdVec[0]);
    SK_LOGD("there exists only one memory write node which is notify, it may cause dead lock, details=%s",
            writeNode->Format().c_str());
    writeNode->SetNodeType(SkNodeType::NODE_NOTIFY);
    writeNode->SetIsFusible(enablePairedWaitBypass);
    if (!AddEventAssociateNotify(eventId, writeNode)) {
      SK_LOGE("Failed to associate notify event 0x%lx with node %lu", eventId, notifyIdVec[0]);
      return false;
    }
    waitFusible = enablePairedWaitBypass;
    if (enablePairedWaitBypass) {
      SK_LOGI(
          "value_breaker_bypass=%d enable 0b01 paired notify/wait bypass, "
          "convert paired event 0x%lx to fusible",
          valueBreakerBypass, eventId);
    }
  } else {
    waitFusible = enableUnpairedWaitBypass;
    if (enableUnpairedWaitBypass) {
      SK_LOGI(
          "value_breaker_bypass=%d enable 0b10 unpaired wait bypass, "
          "convert unpaired wait event 0x%lx to fusible",
          valueBreakerBypass, eventId);
    }
  }

  for (auto waitNodeId : memoryInfo.waitNodeIdList) {
    auto *waitNode = GetNodeById(waitNodeId);
    if (waitNode != nullptr && (waitNode->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT)) {
      waitNode->SetNodeType(SkNodeType::NODE_WAIT);
      waitNode->SetIsFusible(waitFusible);
      if (!AddEventAssociateWait(eventId, waitNode)) {
        SK_LOGE("Failed to associate wait event 0x%lx with node %lu", eventId, waitNodeId);
        return false;
      }
    }
  }
  // Reset nodes preserve event ordering but do not enter fusion.
  if (!resetIdVec.empty()) {
    for (auto resetId : resetIdVec) {
      auto *resetNode = GetNodeById(resetId);
      resetNode->SetNodeType(SkNodeType::NODE_RESET);
      if (!AddEventAssociateReset(eventId, resetNode)) {
        SK_LOGE("Failed to associate reset event 0x%lx with node %lu", eventId, resetId);
        return false;
      }
    }
  }
  return true;
}

bool SuperKernelGraph::PostProcessMemoryNode() {
  SK_LOGI("Starting to build memory node associations, total events: %zu", memoryToNodes.size());
  const uint32_t valueBreakerBypass = GetValueBreakerBypass();
  // value_breaker_bypass is a wait-oriented bitmask
  const bool enableUnpairedWaitBypass = (valueBreakerBypass & ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT) != 0;
  for (const auto &it : memoryToNodes) {
    const uint64_t eventId = it.first;
    const MemoryInfos &memoryInfo = it.second;
    if (memoryInfo.writeNodeIdList.empty() && !memoryInfo.waitNodeIdList.empty()) {
      // No memory write exists, meaning the memory write is outside modelRI.
      if (enableUnpairedWaitBypass) {
        SK_LOGI(
            "value_breaker_bypass=0x%x enables 0b10 unpaired wait bypass, "
            "mark wait event 0x%lx as fusible",
            valueBreakerBypass, eventId);
      }
      for (auto waitNodeId : memoryInfo.waitNodeIdList) {
        auto *waitNode = GetNodeById(waitNodeId);
        if (waitNode != nullptr && (waitNode->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT)) {
          waitNode->SetNodeType(SkNodeType::NODE_WAIT);
          waitNode->SetIsFusible(enableUnpairedWaitBypass);
          if (!AddEventAssociateWait(eventId, waitNode)) {
            SK_LOGE("Failed to associate wait event 0x%lx with node %lu", eventId, waitNodeId);
            return false;
          }
        }
      }
    } else if (!memoryInfo.writeNodeIdList.empty() && memoryInfo.waitNodeIdList.empty()) {
      // only exists memory write nodes, treat them as write-only memory events.
      for (auto writeNodeId : memoryInfo.writeNodeIdList) {
        auto *writeNode = GetNodeById(writeNodeId);
        if (writeNode != nullptr && (writeNode->GetNodeType() == SkNodeType::NODE_MEMORY_WRITE)) {
          const uint64_t writeMemoryValue = ResolveEventValue(writeNode);
          const bool isReset = (writeMemoryValue == SK_DEFAULT_RESET_VALUE);
          const auto targetType = isReset ? SkNodeType::NODE_RESET : SkNodeType::NODE_NOTIFY;
          writeNode->SetNodeType(targetType);
          writeNode->SetIsFusible(true);
          if (isReset) {
            if (!AddEventAssociateReset(eventId, writeNode)) {
              SK_LOGE("Failed to associate reset event 0x%lx with node %lu", eventId, writeNodeId);
              return false;
            }
          } else {
            if (!AddEventAssociateNotify(eventId, writeNode)) {
              SK_LOGE("Failed to associate notify event 0x%lx with node %lu", eventId, writeNodeId);
              return false;
            }
          }
        }
      }
    } else {
      // Check consistency of memoryValue and flag in waitNode list
      std::optional<std::pair<uint64_t, uint32_t>> firstWaitInfo;

      for (auto waitNodeId : memoryInfo.waitNodeIdList) {
        auto *waitNode = GetNodeById(waitNodeId);
        if (waitNode == nullptr || waitNode->GetNodeType() != SkNodeType::NODE_MEMORY_WAIT) {
          SK_LOGE("Invalid wait node found for event 0x%lx, %s", eventId,
                  waitNode ? waitNode->Format().c_str() : "NOT_FOUND");
          return false;
        }

        const uint64_t memoryValue = waitNode->nodeInfos.syncInfos.memoryValue;
        const uint32_t flag = waitNode->nodeInfos.syncInfos.memoryWaitFlag;

        if (!firstWaitInfo.has_value()) {
          firstWaitInfo = {memoryValue, flag};
        } else if (memoryValue != firstWaitInfo->first || flag != firstWaitInfo->second) {
          // Inconsistent wait parameters found, log all waitNode details
          SK_LOGE("waitNode list contains inconsistent memoryValue/flag combinations, eventId: 0x%lx", eventId);
          for (auto waitNodeId : memoryInfo.waitNodeIdList) {
            auto *waitNode = GetNodeById(waitNodeId);
            if (waitNode != nullptr && (waitNode->GetNodeType() == SkNodeType::NODE_MEMORY_WAIT)) {
              SK_LOGE("%s", waitNode->Format().c_str());
            }
          }
          return false;
        }
      }

      if (!firstWaitInfo.has_value()) {
        SK_LOGE("No valid memory wait node found for memory event 0x%lx", eventId);
        return false;
      }
      uint64_t memoryWaitValue = firstWaitInfo->first;
      uint32_t waitFlag = firstWaitInfo->second;

      bool ret = ProcessMemoryWriteNodes(eventId, memoryInfo, memoryWaitValue, waitFlag);
      if (!ret) {
        SK_LOGE("Failed to process paired memory write nodes, eventId: 0x%lx", eventId);
        return ret;
      }
    }
  }
  SK_LOGI("Completed building all memory node associations");
  return true;
}

namespace {

// Scope stack entry for tracking nested scope contexts
struct ScopeStackEntry {
  uint32_t scopeIdx = INVALID_SCOPE_ID;
  std::string scopeName;
  bool isFusible = true;
  FusionFailReasonInfo failReason;
};

// Compute current scope bit flags from the scope stack
// Only marks fusible scopes to determine which fusible scopes a node belongs to
std::bitset<MAX_SCOPE_NUM> ComputeScopeBitFlags(const std::vector<ScopeStackEntry> &scopeStack) {
  std::bitset<MAX_SCOPE_NUM> flags;
  for (const auto &entry : scopeStack) {
    if (entry.isFusible) {
      flags.set(entry.scopeIdx);
    }
  }
  return flags;
}

FusionFailReasonInfo GetUnfusibleScopeFailReason(const std::vector<ScopeStackEntry> &scopeStack) {
  for (auto iter = scopeStack.rbegin(); iter != scopeStack.rend(); ++iter) {
    if (iter->isFusible) {
      continue;
    }
    return iter->failReason;
  }
  return FusionFailReasonInfo();
}

// Parse scope node information to extract scope index, name, and fusible attribute
// Get scope index for a scope node
// For fusible scopes, look up index from scopeNameToIdx; for unfusible scopes, use MAX_SCOPE_NUM
uint32_t GetScopeIdx(SuperKernelBaseNode *node, const std::unordered_map<std::string, uint32_t> &scopeNameToIdx) {
  bool isFusible = node->IsFusible();

  if (isFusible) {
    const std::string &scopeName = node->GetScopeName();
    auto it = scopeNameToIdx.find(scopeName);
    if (it == scopeNameToIdx.end()) {
      SK_LOGE(
          "Fusible scope name '%s' not registered for node %lu (node marked fusible but scope not in registry), graph "
          "structure error",
          scopeName.c_str(), node->GetNodeId());
      return MAX_SCOPE_NUM;
    }
    return it->second;
  }

  // Unfusible scopes use MAX_SCOPE_NUM to avoid conflict with fusible scope indices [0, MAX_SCOPE_NUM-1]
  // The actual index value is not used since unfusible scopes are excluded from scope bit flags
  return MAX_SCOPE_NUM;
}

// Pop the scope with matching name from the stack
// Searches from top to bottom to find the matching scope and removes only that scope
// This supports interleaved scope begin/end patterns (e.g., A-B-A-B)
bool PopScopeByName(std::vector<ScopeStackEntry> &scopeStack, const std::string &scopeName) {
  for (int i = static_cast<int>(scopeStack.size()) - 1; i >= 0; --i) {
    if (scopeStack[i].scopeName == scopeName) {
      scopeStack.erase(scopeStack.begin() + i);
      SK_LOGI("Popped scope '%s' at position %d, stack_size=%zu", scopeName.c_str(), i, scopeStack.size());
      return true;
    }
  }
  SK_LOGE("Scope end without matching begin: name='%s', graph structure error (missing corresponding scope begin)",
          scopeName.c_str());
  return false;
}

// Process scope begin node: parse scope info, push to stack, and mark associated event nodes
void ProcessScopeBegin(SuperKernelGraph *graph, SuperKernelBaseNode *node, std::vector<ScopeStackEntry> &scopeStack,
                       const std::unordered_map<std::string, uint32_t> &scopeNameToIdx) {
  std::string scopeName = node->GetScopeName();
  bool isFusible = node->IsFusible();
  uint32_t scopeIdx = GetScopeIdx(node, scopeNameToIdx);
  FusionFailReasonInfo failReason = isFusible ? FusionFailReasonInfo() : node->GetFusionFailReasonInfo();
  if (!isFusible && failReason == FusionFailReason::CAN_FUSE) {
    failReason = FusionFailReasonInfo(FusionFailReason::IN_UNFUSIBLE_SCOPE);
  }

  scopeStack.push_back({scopeIdx, scopeName, isFusible, failReason});
  SK_LOGI("Scope begin: name='%s' idx=%u fusible=%d stack_size=%zu", scopeName.c_str(), scopeIdx, isFusible,
          scopeStack.size());
}

// Process scope end node: parse scope info, pop from stack, and mark associated event nodes
bool ProcessScopeEnd(SuperKernelGraph *graph, SuperKernelBaseNode *node, std::vector<ScopeStackEntry> &scopeStack,
                     const std::unordered_map<std::string, uint32_t> &scopeNameToIdx) {
  std::string scopeName = node->GetScopeName();
  uint32_t scopeIdx = GetScopeIdx(node, scopeNameToIdx);

  SK_LOGI("Scope end: name='%s' idx=%u", scopeName.c_str(), scopeIdx);
  return PopScopeByName(scopeStack, scopeName);
}

// Log warning if there are unclosed scopes remaining in the stack at the end of graph processing
void LogUnclosedScopes(const std::vector<ScopeStackEntry> &scopeStack) {
  if (!scopeStack.empty()) {
    SK_LOGE("Found %zu unclosed scope(s) at end of graph: graph structure error (missing scope end nodes)",
            scopeStack.size());
    for (const auto &entry : scopeStack) {
      SK_LOGE("  - Scope '%s' (idx=%u, fusible=%d)", entry.scopeName.c_str(), entry.scopeIdx, entry.isFusible);
    }
  }
}

}  // namespace

// Update scope bit flags for all nodes based on scope contexts
//
// The algorithm processes nodes in sorted order and maintains a scope stack to track nested scopes:
// 1. When encountering a scope begin node, push scope info to stack
// 2. For scope end nodes, compute flags BEFORE popping (node belongs to the scope it closes), then pop
// 3. For all other nodes (scope begin, notify, wait, regular nodes), compute flags from current stack
//
// Scope bit flags indicate which fusible scopes a node belongs to, enabling scope-based fusion decisions
//
// Node types:
// - Scope begin kernel node: marks start of a scope, also belongs to its parent scopes
// - Scope end kernel node: marks end of a scope, belongs to its parent scopes
// - Notify/Wait nodes: scope-related synchronization nodes, not scope begin/end
// - Regular kernel nodes: normal computation nodes
//
// Note: Scopes with the same name follow stack semantics (LIFO), allowing nested and sequential scopes
bool SuperKernelGraph::UpdateNodeScopeBitFlags() {
  std::vector<ScopeStackEntry> scopeStack;
  std::vector<uint64_t> orderedNodeIds = GetSortedNodeIds();

  SK_LOGI("Starting UpdateNodeScopeBitFlags, total nodes: %zu", orderedNodeIds.size());
  bool outOfScopeFusible = true;
  if (!scopeNameToIdx.empty()) {
    outOfScopeFusible = false;
    SK_LOGI("Super Kernel has named scopes, out-of-scope nodes will be marked as unfusible");
  }
  for (uint64_t nodeId : orderedNodeIds) {
    SuperKernelBaseNode *node = GetNodeById(nodeId);
    if (node == nullptr) {
      SK_LOGE("Node with id %lu not found", nodeId);
      continue;
    }

    if (node->IsScopeBegin()) {
      ProcessScopeBegin(this, node, scopeStack, scopeNameToIdx);
    } else if (node->IsScopeEnd()) {
      // Scope end nodes belong to their parent scopes, compute flags before popping
      std::bitset<MAX_SCOPE_NUM> currentScopeFlags = ComputeScopeBitFlags(scopeStack);
      node->SetScopeBitFlags(currentScopeFlags);
      if (!ProcessScopeEnd(this, node, scopeStack, scopeNameToIdx)) {
        return false;
      }
    }

    // Update flags for all nodes except scope end nodes (already handled above)
    // This includes: scope begin nodes, notify nodes, wait nodes, and regular kernel nodes
    if (!node->IsScopeEnd()) {
      std::bitset<MAX_SCOPE_NUM> flags = ComputeScopeBitFlags(scopeStack);
      node->SetScopeBitFlags(flags);

      if (!outOfScopeFusible && scopeStack.empty()) {
        // If there are named scopes, mark nodes outside of any scope as unfusible
        node->SetIsFusible(false);
        node->SetFusionFailReason(FusionFailReason::NOT_IN_SCOPE);
        SK_LOGI("Marked node %s as unfusible (outside of any named scope)", node->Format().c_str());
      }
      // Mark regular nodes as unfusible if inside any unfusible scope
      FusionFailReasonInfo scopeFailReason = GetUnfusibleScopeFailReason(scopeStack);
      if (!node->IsScopeNode() && scopeFailReason != FusionFailReason::CAN_FUSE) {
        node->SetIsFusible(false);
        node->SetFusionFailReason(scopeFailReason);
        SK_LOGI("Marked node %s as unfusible (inside unfusible scope, reason=%s)", node->Format().c_str(),
                FusionFailReasonToStr(scopeFailReason).c_str());
      }
    }

    // scope nodes are always marked as fusible
    if (node->IsScopeNode()) {
      node->SetIsFusible(true);
      needUpdateNodes.insert(node);
    }
    {
      SK_LOG_CONTEXT_SIMPLE("sk_node_detail.log");
      SK_LOGI("Processed node %s: type=%s, scopeFlags=%s, isFusible=%d, stackSize=%zu", node->Format().c_str(),
              to_string(node->GetNodeType()), BitsetToString(node->GetScopeBitFlags()).c_str(), node->IsFusible(),
              scopeStack.size());
    }
    SK_LOGI("Processed node %s: type=%s, scopeFlags=%s, isFusible=%d, stackSize=%zu", node->Format().c_str(),
            to_string(node->GetNodeType()), BitsetToString(node->GetScopeBitFlags()).c_str(), node->IsFusible(),
            scopeStack.size());
  }
  LogUnclosedScopes(scopeStack);
  if (!scopeStack.empty()) {
    return false;
  }
  SK_LOGI("UpdateNodeScopeBitFlags completed");
  return true;
}

namespace {
// ============ ParseOriginalScopes Helper Functions ============

struct ScopeNodeTypes {
  bool hasNamedScopeNode = false;
  bool hasUnnamedScopeNode = false;
};

ScopeNodeTypes CheckScopeNodeTypes(const std::vector<uint64_t> &orderedNodeIds, SuperKernelGraph *graph) {
  ScopeNodeTypes types;
  for (uint64_t nodeId : orderedNodeIds) {
    SuperKernelBaseNode *node = graph->GetNodeById(nodeId);
    if (node == nullptr || !node->IsScopeNode() || !node->IsScopeBegin()) {
      continue;
    }
    if (node->GetScopeName().empty()) {
      types.hasUnnamedScopeNode = true;
    } else {
      types.hasNamedScopeNode = true;
    }
  }
  return types;
}

bool IsValidNodeForScope(SuperKernelBaseNode *node) {
  if (node == nullptr) return false;
  if (node->IsScopeBegin() || node->IsScopeEnd() || node->IsScopePlaceholder()) {
    return false;
  }
  return !node->GetScopeBitFlags().none();
}

void GroupNodesByScopeBitFlags(const std::vector<uint64_t> &orderedNodeIds, SuperKernelGraph *graph,
                               std::vector<OriginalScopeInfo> &scopeInfos) {
  std::unordered_map<std::bitset<MAX_SCOPE_NUM>, OriginalScopeInfo> scopeGroups;

  for (uint64_t nodeId : orderedNodeIds) {
    SuperKernelBaseNode *node = graph->GetNodeById(nodeId);
    if (!IsValidNodeForScope(node)) {
      continue;
    }
    std::bitset<MAX_SCOPE_NUM> flags = node->GetScopeBitFlags();
    scopeGroups[flags].nodeIds.push_back(nodeId);
    scopeGroups[flags].scopeBitFlags = flags;
  }

  for (auto &pair : scopeGroups) {
    pair.second.scopeId = static_cast<uint16_t>(scopeInfos.size());
    scopeInfos.push_back(std::move(pair.second));
  }
}

void CreateSingleScopeFromAllNodes(const std::vector<uint64_t> &orderedNodeIds, SuperKernelGraph *graph,
                                   std::vector<OriginalScopeInfo> &scopeInfos) {
  OriginalScopeInfo info;
  info.scopeId = 0;

  for (uint64_t nodeId : orderedNodeIds) {
    SuperKernelBaseNode *node = graph->GetNodeById(nodeId);
    if (node == nullptr) {
      continue;
    }
    info.nodeIds.push_back(nodeId);
    info.scopeBitFlags = node->GetScopeBitFlags();
  }

  if (!info.nodeIds.empty()) {
    scopeInfos.push_back(std::move(info));
  }
}

void CreateScopesExcludingUnnamed(const std::vector<uint64_t> &orderedNodeIds, SuperKernelGraph *graph,
                                  std::vector<OriginalScopeInfo> &scopeInfos) {
  OriginalScopeInfo info;
  info.scopeId = 0;
  bool insideUnnamedScope = false;

  for (uint64_t nodeId : orderedNodeIds) {
    SuperKernelBaseNode *node = graph->GetNodeById(nodeId);
    if (node == nullptr) {
      continue;
    }

    if (node->IsScopeBegin()) {
      if (node->GetScopeName().empty()) {
        insideUnnamedScope = true;
      }
      continue;
    }

    if (node->IsScopeEnd()) {
      if (insideUnnamedScope) {
        insideUnnamedScope = false;
      }
      continue;
    }

    if (node->IsScopePlaceholder() || insideUnnamedScope) {
      continue;
    }

    info.nodeIds.push_back(nodeId);
    info.scopeBitFlags = node->GetScopeBitFlags();
  }

  if (!info.nodeIds.empty()) {
    scopeInfos.push_back(std::move(info));
  }
}

}  // namespace

void SuperKernelGraph::ParseOriginalScopes() {
  originalScopeInfos_.clear();

  std::vector<uint64_t> orderedNodeIds = GetSortedNodeIds();
  ScopeNodeTypes scopeTypes = CheckScopeNodeTypes(orderedNodeIds, this);

  SK_LOGI("ParseOriginalScopes: hasNamedScopeNode=%d, hasUnnamedScopeNode=%d", scopeTypes.hasNamedScopeNode,
          scopeTypes.hasUnnamedScopeNode);

  // Case 1: Has named scope nodes - group by scopeBitFlags
  if (scopeTypes.hasNamedScopeNode) {
    SK_LOGI("ParseOriginalScopes: case 1 - has named scope nodes, group by scopeBitFlags");
    GroupNodesByScopeBitFlags(orderedNodeIds, this, originalScopeInfos_);
  }
  // Case 2: No scope nodes at all - all nodes as one scope
  else if (!scopeTypes.hasUnnamedScopeNode) {
    SK_LOGI("ParseOriginalScopes: case 2 - no scope nodes, all nodes as one scope");
    CreateSingleScopeFromAllNodes(orderedNodeIds, this, originalScopeInfos_);
  }
  // Case 3: Only unnamed scope nodes - exclude nodes inside unnamed scopes
  else {
    SK_LOGI("ParseOriginalScopes: case 3 - only unnamed scope nodes, exclude nodes inside unnamed scopes");
    CreateScopesExcludingUnnamed(orderedNodeIds, this, originalScopeInfos_);
  }

  SK_LOGI("ParseOriginalScopes completed: %zu original scopes", originalScopeInfos_.size());
}

std::unique_ptr<SuperKernelBaseNode> SuperKernelNodeFactory::CreateNode(std::unique_ptr<aclmdlRITask> task,
                                                                        aclmdlRITaskType taskType, uint64_t nodeIdx,
                                                                        uint64_t streamIdxInGraph, int32_t streamId,
                                                                        uint64_t preNodeId) {
  switch (taskType) {
    case ACL_MODEL_RI_TASK_KERNEL:
      return std::make_unique<SuperKernelKernelNode>(std::move(task), taskType, nodeIdx, streamIdxInGraph, streamId,
                                                     preNodeId);
    case ACL_MODEL_RI_TASK_EVENT_RECORD:
    case ACL_MODEL_RI_TASK_EVENT_WAIT:
    case ACL_MODEL_RI_TASK_EVENT_RESET:
    case ACL_MODEL_RI_TASK_VALUE_WRITE:
    case ACL_MODEL_RI_TASK_VALUE_WAIT:
      return std::make_unique<SuperKernelMemoryNode>(std::move(task), taskType, nodeIdx, streamIdxInGraph, streamId,
                                                     preNodeId);
    default:
      return std::make_unique<SuperKernelDefaultNode>(std::move(task), taskType, nodeIdx, streamIdxInGraph, streamId,
                                                      preNodeId);
  }
}

bool SuperKernelGraph::InitSKGraph() {
  SK_LOGI("Starting to initialize SuperKernel graph");

  SK_LOGI("current model id: %s", modelId.c_str());

  if (!InitFromModelRI()) {
    return false;
  }

  SK_LOGI("Total nodes added: %zu, total streams: %zu", graphMap.size(), streams.size());
  SK_LOGI("Starting UpdateNodeScopeBitFlags");
  if (!UpdateNodeScopeBitFlags()) {
    SK_LOGE("Failed to update node scope bit flags: scope begin and end nodes are not matched");
    return false;
  }
  SK_LOGI("UpdateNodeScopeBitFlags completed");

  SK_LOGI("Starting ParseOriginalScopes");
  ParseOriginalScopes();
  SK_LOGI("ParseOriginalScopes completed");

  SK_LOGI("Starting PostProcessMemoryNode");
  bool flag = PostProcessMemoryNode();
  SK_LOGI("PostProcessMemoryNode completed");
  if (!flag) {
    return flag;
  }

  SK_LOGI("Starting BuildEventNodeAssociations");
  BuildEventNodeAssociations();
  SK_LOGI("BuildEventNodeAssociations completed");

  SK_LOGI("Starting AddEventAssociate");
  AddEventAssociate();
  SK_LOGI("AddEventAssociate completed");

  SK_LOGI("Successfully initialized SuperKernel graph with %zu nodes and %zu streams", graphMap.size(), streams.size());

  return true;
}

/**
 * @brief Initialize graph from modelRI (encapsulated method)
 *
 * This function initializes the SuperKernel graph by:
 * 1. Initializing and getting all streams from modelRI
 * 2. Processing all streams and tasks
 *
 * @return true Initialization successful
 * @return false Initialization failed
 */
bool SuperKernelGraph::InitFromModelRI() {
  SK_LOGI("Starting to initialize SuperKernel graph from modelRI");

  // Step 1: Initialize and get all streams
  std::vector<uint32_t> streamTaskNums;
  if (!InitStreamsFromModelRI(streamTaskNums)) {
    return false;
  }

  // Step 2: Process all streams and tasks
  if (!ProcessAllStreamsAndTasks(streamTaskNums)) {
    return false;
  }

  SK_LOGI("Successfully initialized SuperKernel graph from modelRI with %zu nodes and %zu streams", graphMap.size(),
          streams.size());

  return true;
}

/**
 * @brief Initialize streams from modelRI
 *
 * Gets the number of streams from modelRI, allocates storage, and retrieves all stream handles.
 *
 * @return true Stream initialization successful
 * @return false Stream initialization failed
 */
bool SuperKernelGraph::InitStreamsFromModelRI(std::vector<uint32_t> &streamTaskNums) {
  uint32_t streamNum = 0;
  aclError ret = aclmdlRIGetStreams(modelRI, nullptr, &streamNum);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get number of streams in model runtime image, ret=%d", ret);
    return false;
  }
  SK_LOGI("Get %u streams from model runtime image", streamNum);

  std::vector<aclrtStream> modelStreams(streamNum);
  ret = aclmdlRIGetStreams(modelRI, modelStreams.data(), &streamNum);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get streams in model runtime image, ret=%d", ret);
    return false;
  }

  std::vector<aclrtStream> validStreams;
  std::vector<uint32_t> validStreamTaskNums;
  validStreams.reserve(streamNum);
  validStreamTaskNums.reserve(streamNum);
  for (uint32_t streamIdx = 0; streamIdx < streamNum; ++streamIdx) {
    uint32_t taskNum = 0;
    ret = aclmdlRIGetTasksByStream(modelStreams[streamIdx], nullptr, &taskNum);
    if (ret != ACL_SUCCESS) {
      SK_LOGE("Failed to get number of tasks in stream %u, ret=%d", streamIdx, ret);
      return false;
    }
    if (taskNum == 0) {
      SK_LOGI("No tasks found in stream %u, skip this stream", streamIdx);
      continue;
    }
    validStreams.emplace_back(modelStreams[streamIdx]);
    validStreamTaskNums.emplace_back(taskNum);
  }

  streams = std::move(validStreams);
  streamTaskNums = std::move(validStreamTaskNums);
  return true;
}

/**
 * @brief Process all streams and tasks
 *
 * Iterates through all streams, gets tasks from each stream, and processes them.
 * @param streamTaskNums Number of tasks in each stream
 *
 * @return true Processing successful
 * @return false Processing failed
 */
bool SuperKernelGraph::ProcessAllStreamsAndTasks(const std::vector<uint32_t> &streamTaskNums) {
  if (streamTaskNums.empty()) {
    SK_LOGI("No tasks found in model runtime image, skip processing streams and tasks");
    return true;
  }

  const uint32_t streamNum = static_cast<uint32_t>(streams.size());
  uint32_t maxTaskNum = *std::max_element(streamTaskNums.begin(), streamTaskNums.end());
  auto tasks = std::make_unique<aclmdlRITask[]>(maxTaskNum);

  for (uint32_t streamIdx = 0; streamIdx < streamNum; ++streamIdx) {
    uint32_t taskNum = streamTaskNums[streamIdx];
    aclError ret = aclmdlRIGetTasksByStream(streams[streamIdx], tasks.get(), &taskNum);
    if (ret != ACL_SUCCESS) {
      SK_LOGE("Failed to get tasks in stream %u, ret=%d", streamIdx, ret);
      return false;
    }
    nodeSizeInStream.emplace_back(taskNum);
    SK_LOGI("Stream %u has %u tasks", streamIdx, taskNum);

    uint64_t preNodeId = INVALID_TASK_ID;
    for (uint32_t taskIdx = 0; taskIdx < taskNum; ++taskIdx) {
      if (!ProcessSingleTask(tasks[taskIdx], streamIdx, taskIdx, preNodeId)) {
        SK_LOGE("Failed to process task %u in stream %u", taskIdx, streamIdx);
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Process single task
 *
 * Creates a node from the task, initializes it, registers fusible scope,
 * adds it to the graph, and updates node relationships.
 *
 * @param task The task to process
 * @param streamIdx Stream index
 * @param taskIdx Task index within the stream
 * @param preNodeId Previous node ID (updated to current node ID after processing)
 * @return true Task processing successful
 * @return false Task processing failed
 */
bool SuperKernelGraph::ProcessSingleTask(aclmdlRITask &task, uint32_t streamIdx, uint32_t taskIdx,
                                         uint64_t &preNodeId) {
  aclmdlRITaskType taskType;
  aclError ret = aclmdlRITaskGetType(task, &taskType);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get task type for task %u in stream %u, ret=%d", taskIdx, streamIdx, ret);
    return false;
  }

  int32_t realStreamId = -1;
  ret = aclrtStreamGetId(streams[streamIdx], &realStreamId);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("Failed to get stream ID for stream %u, ret=%d", streamIdx, ret);
    return false;
  }

  auto node = SuperKernelNodeFactory::CreateNode(std::make_unique<aclmdlRITask>(task), taskType, taskIdx, streamIdx,
                                                 realStreamId, preNodeId);
  if (!node->InitNode(opts_)) {
    SK_LOGE("Failed to initialize node for task %u in stream %u (taskType=%u, nodeId=%lu)", taskIdx, streamIdx,
            taskType, node->GetNodeId());
    return false;
  }

  RegisterFusibleScope(node);

  uint64_t nodeId = node->GetNodeId();
  if (!AddNode(std::move(node))) {
    SK_LOGE("Failed to add node for task %u in stream %u to graph (likely duplicate node ID %lu)", taskIdx, streamIdx,
            nodeId);
    return false;
  }

  UpdateNodeRelations(nodeId, streamIdx, taskIdx, preNodeId);
  return true;
}

/**
 * @brief Register fusible scope
 *
 * Registers a fusible scope if the node is a kernel node with a valid scope name.
 * Maps scope name to a unique index for later reference.
 *
 * @param node The node to register scope from
 */
void SuperKernelGraph::RegisterFusibleScope(const std::unique_ptr<SuperKernelBaseNode> &node) {
  if (node->GetNodeType() == SkNodeType::NODE_KERNEL && node->IsScopeNode()) {
    if (node->GetScopeName().length() > 0 && node->IsFusible()) {
      const auto &scopeName = node->GetScopeName();
      if (scopeNameToIdx.find(scopeName) != scopeNameToIdx.end()) {
        return;
      }
      if (scopeNameToIdx.size() >= MAX_SCOPE_NUM) {
        SK_LOGW("Exceeded maximum scope limit %u, marking scope '%s' as unfusible", MAX_SCOPE_NUM, scopeName.c_str());
        node->SetIsFusible(false);
        node->SetFusionFailReason(FusionFailReason::EXCEED_SCOPE_MAX);
        return;
      }
      uint32_t scopeIdx = static_cast<uint32_t>(scopeNameToIdx.size());
      scopeNameToIdx[scopeName] = scopeIdx;
      scopeIdxToName[scopeIdx] = scopeName;
      SK_LOGI("Registered fusible scope '%s' with index %u", scopeName.c_str(), scopeIdx);
    }
  }
}

/**
 * @brief Update node relationships
 *
 * Updates the graph structure by:
 * 1. Adding the node to headNodes if it's the first task in the stream
 * 2. Setting the next node ID for the previous node
 * 3. Updating preNodeId to current node ID
 *
 * @param nodeId Current node ID
 * @param streamIdx Stream index
 * @param taskIdx Task index within the stream
 * @param preNodeId Previous node ID (updated to current node ID)
 */
void SuperKernelGraph::UpdateNodeRelations(uint64_t nodeId, uint32_t streamIdx, uint32_t taskIdx, uint64_t &preNodeId) {
  if (taskIdx == 0) {
    headNodes.push_back(nodeId);
    SK_LOGI("Stream %u: Added head node %lu", streamIdx, nodeId);
  }
  if (preNodeId != INVALID_TASK_ID) {
    graphMap[preNodeId]->SetNextNodeId(nodeId);
  }
  preNodeId = nodeId;
}

aclrtStream SuperKernelGraph::GetStreamByIndex(uint32_t streamIdx) const {
  if (streamIdx >= streams.size()) {
    SK_LOGE("Stream index %u out of bounds, total streams: %zu", streamIdx, streams.size());
    return nullptr;
  }
  return streams[streamIdx];
}

std::vector<uint64_t> SuperKernelGraph::GetSortedNodeIds() const {
  std::vector<uint64_t> nodeIds;
  nodeIds.reserve(graphMap.size());
  for (const auto &pair : graphMap) {
    nodeIds.push_back(pair.first);
  }
  std::sort(nodeIds.begin(), nodeIds.end());
  return nodeIds;
}

SuperKernelGraph::FusionFailStats SuperKernelGraph::CollectFusionFailStats() {
  FusionFailStats stats;
  std::vector<uint64_t> sortedNodeIds = GetSortedNodeIds();

  for (uint64_t nodeId : sortedNodeIds) {
    SuperKernelBaseNode *node = GetNodeById(nodeId);
    if (node == nullptr) {
      continue;
    }

    const FusionFailReasonInfo &reasonInfo = node->fusionFailReason_;
    bool isFusible = node->IsFusible();
    const std::string reasonKey = FusionFailReasonToStr(reasonInfo);

    // Update statistics
    if (isFusible) {
      stats.fusibleCount++;
    } else {
      stats.unfusibleCount++;
      stats.reasonStats[reasonKey]++;
    }

    std::string reasonForLog = reasonKey;
    const std::string reasonDetail = FusionFailReasonDetailToStr(reasonInfo);
    if (!reasonDetail.empty()) {
      reasonForLog += ", reasonDetail: " + reasonDetail;
    }

    // Collect node log entry for all nodes (for plog)
    std::string logEntry =
        "Node " + node->Format() + ": isFusible: " + std::to_string(isFusible) + ", reason: " + reasonForLog;
    stats.nodeLogEntries.push_back(std::move(logEntry));

    // Collect unfusible KERNEL node log entry only (for log file)
    if (!isFusible && node->GetNodeType() == SkNodeType::NODE_KERNEL && !node->IsScopeNode()) {
      stats.unfusibleNodeLogEntries.push_back("Node " + node->Format() + ": reason: " + reasonForLog);
    }
  }

  return stats;
}

void SuperKernelGraph::DumpFusionFailReasons(const std::vector<SuperKernelScopeInfo> &processedScopeInfos,
                                             bool debugPerOpMaxCoreEnabled) {
  SK_LOGI("Starting to dump fusion fail reasons for all nodes");

  // Collect statistics
  FusionFailStats stats;
  stats = CollectFusionFailStats();

  // Output unfusible nodes to log file (sk_fusion_fail_reasons.log)
  {
    SK_LOG_CONTEXT_SIMPLE("sk_fusion_fail_reasons.log");
    SK_LOGI("unfusible kernel:");
    for (const auto &entry : stats.unfusibleNodeLogEntries) {
      SK_LOGI("%s", entry.c_str());
    }

    // Print original scopes before fusion (from ParseOriginalScopes)
    PrintOriginalScopes(*this);

    // Print scopes after fusion
    PrintFusedScopes(*this, processedScopeInfos, debugPerOpMaxCoreEnabled);
  }  // End of SK_LOG_CONTEXT_SIMPLE block

  // Output unfusible KERNEL nodes to plog (same as log file)
  for (const auto &entry : stats.unfusibleNodeLogEntries) {
    SK_LOGI("%s", entry.c_str());
  }

  // Print original scope to fused scope mapping
  SK_LOGI("Original Scope to Fused Scope Mapping:");
  PrintOriginalScopes(*this);
  PrintFusedScopes(*this, processedScopeInfos, debugPerOpMaxCoreEnabled);

  // Print summary statistics
  SK_LOGI("Fusion fail reasons summary:");
  SK_LOGI("  Total nodes: %zu", graphMap.size());
  SK_LOGI("  Fusible nodes: %zu", stats.fusibleCount);
  SK_LOGI("  Unfusible nodes: %zu", stats.unfusibleCount);

  if (stats.unfusibleCount > 0) {
    SK_LOGI("  Failure reason breakdown:");
    for (const auto &pair : stats.reasonStats) {
      SK_LOGI("    %s: %zu nodes", pair.first.c_str(), pair.second);
    }
  }

  SK_LOGI("Completed dumping fusion fail reasons");
}

// Forward declarations for node ToJson functions
Json SuperKernelKernelNodeToJson(const SuperKernelKernelNode *node);
Json SuperKernelMemoryNodeToJson(const SuperKernelMemoryNode *node);
Json SuperKernelDefaultNodeToJson(const SuperKernelDefaultNode *node);

/**
 * @brief Convert a single node to JSON based on its type
 */
static Json NodeToJsonByType(const SuperKernelBaseNode *node) {
  if (node == nullptr) {
    return Json();
  }

  switch (node->GetNodeType()) {
    case SkNodeType::NODE_KERNEL:
      return SuperKernelKernelNodeToJson(static_cast<const SuperKernelKernelNode *>(node));
    case SkNodeType::NODE_NOTIFY:
    case SkNodeType::NODE_WAIT:
    case SkNodeType::NODE_RESET:
    case SkNodeType::NODE_MEMORY_WRITE:
    case SkNodeType::NODE_MEMORY_WAIT:
      return SuperKernelMemoryNodeToJson(static_cast<const SuperKernelMemoryNode *>(node));
    case SkNodeType::NODE_DEFAULT:
    default:
      return SuperKernelDefaultNodeToJson(static_cast<const SuperKernelDefaultNode *>(node));
  }
}

static Json NodeWrapperToJson(const SuperKernelBaseNode *node) {
  Json nodeJson;
  if (node == nullptr) {
    return nodeJson;
  }

  nodeJson["nodeId"] = node->GetNodeId();
  nodeJson["streamIdxInGraph"] = node->GetStreamIdxInGraph();
  nodeJson["nodeIdxInStream"] = node->GetNodeIdxInStream();
  nodeJson["nodeType"] = to_string(node->GetNodeType());
  const std::string scopeName = node->GetScopeName();
  nodeJson["scopeName"] = scopeName.empty() ? Json() : Json(scopeName);
  nodeJson["isScopeNode"] = node->IsScopeNode();
  nodeJson["isScopeBegin"] = node->IsScopeBegin();
  nodeJson["isScopeEnd"] = node->IsScopeEnd();
  nodeJson["isScopePlaceholder"] = node->IsScopePlaceholder();

  const uint64_t preNodeId = node->GetPreNodeId();
  nodeJson["preNodeId"] = (preNodeId == INVALID_TASK_ID) ? Json("none") : Json(preNodeId);
  const uint64_t nextNodeId = node->GetNextNodeId();
  nodeJson["nextNodeId"] = (nextNodeId == INVALID_TASK_ID) ? Json("none") : Json(nextNodeId);

  nodeJson["taskInfo"] = Json::array({NodeToJsonByType(node)});
  return nodeJson;
}

/**
 * @brief Collect all nodes grouped by streams to JSON
 */
static Json CollectStreamsToJson(const SuperKernelGraph &graph) {
  Json streamsJson = Json::array();
  const auto &streams = graph.GetStreams();
  const auto &nodeSizeInStream = graph.GetNodeSizeInStream();

  for (size_t streamIdx = 0; streamIdx < streams.size(); ++streamIdx) {
    Json streamJson;

    streamJson["streamIdxInGraph"] = streamIdx;
    streamJson["nodeCount"] = (streamIdx < nodeSizeInStream.size()) ? nodeSizeInStream[streamIdx] : 0;

    // Collect nodes for this stream. Currently each node maps to one raw task.
    Json nodesJson = Json::array();
    std::vector<uint64_t> sortedNodeIds = graph.GetSortedNodeIds();
    for (uint64_t nodeId : sortedNodeIds) {
      const SuperKernelBaseNode *node = graph.GetNodeById(nodeId);
      if (node != nullptr && static_cast<uint32_t>(streamIdx) == node->GetStreamIdxInGraph()) {
        nodesJson.push_back(NodeWrapperToJson(node));
      }
    }
    streamJson["nodes"] = nodesJson;

    streamsJson.push_back(streamJson);
  }

  return streamsJson;
}

Json SuperKernelGraph::ToJson() const {
  Json rootJson;
  rootJson["version"] = "1.0";
  rootJson["description"] = "SuperKernel Raw Task Information from modelRI";
  rootJson["modelId"] = modelId;

  // Get device ID
  int32_t deviceId = 0;
  aclrtGetDevice(&deviceId);
  rootJson["deviceId"] = deviceId;

  // Options (empty for now, can be extended)
  rootJson["options"]["numOptions"] = 0;
  rootJson["options"]["options"] = Json::array();

  // Total counts
  rootJson["totalStreams"] = streams.size();
  size_t totalNodes = 0;
  for (const auto &pair : graphMap) {
    if (pair.second != nullptr) {
      totalNodes++;
    }
  }
  rootJson["totalNodes"] = totalNodes;

  // Streams with tasks
  rootJson["streams"] = CollectStreamsToJson(*this);

  return rootJson;
}
