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
 * \file sk_graph.h
 * \brief
 */

#ifndef __SK_GRAPH_H__
#define __SK_GRAPH_H__

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>

#include <nlohmann/json.hpp>

#include "sk_log.h"
#include "sk_types.h"
#include "sk_node.h"
#include "sk_scope_info.h"
#include "acl/acl.h"

class SuperKernelOptionsManager;
class ScopeVerifyGraphBuilder;

using Json = nlohmann::ordered_json;

class SuperKernelNodeFactory {
 public:
  static std::unique_ptr<SuperKernelBaseNode> CreateNode(std::unique_ptr<aclmdlRITask> task, aclmdlRITaskType taskType,
                                                         uint64_t nodeIdx, uint64_t streamIdxInGraph, int32_t streamId,
                                                         uint64_t preNodeId);
};

struct EventInfos {
  uint64_t notifyNodeId = INVALID_TASK_ID;
  std::unordered_set<uint64_t> resetNodeIdList;
  std::unordered_set<uint64_t> waitNodeIdList;
};

struct MemoryInfos {
  std::unordered_set<uint64_t> writeNodeIdList;
  std::unordered_set<uint64_t> waitNodeIdList;
};

class SuperKernelGraph {
 public:
  aclError Update();

  SuperKernelGraph() = default;
  ~SuperKernelGraph() = default;
  SuperKernelGraph(const SuperKernelGraph &) = delete;
  SuperKernelGraph &operator=(const SuperKernelGraph &) = delete;
  SuperKernelGraph(SuperKernelGraph &&) = default;
  SuperKernelGraph &operator=(SuperKernelGraph &&) = default;
  SuperKernelGraph(aclmdlRI modelRI, std::string modelId = {}) : modelRI(modelRI), modelId(std::move(modelId)) {}
  SuperKernelGraph(aclmdlRI modelRI, const SuperKernelOptionsManager &opts, std::string modelId = {})
      : modelRI(modelRI), modelId(std::move(modelId)), opts_(&opts) {}
  bool InitSKGraph();

  SuperKernelBaseNode *GetNodeById(uint64_t nodeId) const;
  aclmdlRI GetModelRI() const {
    return modelRI;
  }

  const std::string &GetModelId() const {
    return modelId;
  }
  const std::vector<uint64_t> &GetHeadNodes() const {
    return headNodes;
  }
  const std::vector<uint64_t> &GetNodeSizeInStream() const {
    return nodeSizeInStream;
  }
  const std::vector<aclrtStream> &GetStreams() const {
    return streams;
  }

  aclrtStream GetStreamByIndex(uint32_t streamIdx) const;

  // Get all nodeIds sorted in ascending order
  std::vector<uint64_t> GetSortedNodeIds() const;

  // Get event info by event id
  const EventInfos *GetEventInfo(uint64_t eventId) const {
    auto it = eventToNodes.find(eventId);
    return it != eventToNodes.end() ? &it->second : nullptr;
  }

  // Get scope name by scope index
  bool GetScopeNameByIdx(uint32_t scopeIdx, std::string &scopeName) const {
    auto it = scopeIdxToName.find(scopeIdx);
    if (it != scopeIdxToName.end()) {
      scopeName = it->second;
      return true;
    }
    return false;
  }

  // Add shape info memory block, managed by graph lifecycle
  void AddShapeInfoPtr(std::unique_ptr<uint8_t[]> ptr) {
    shapeInfoPtrList.emplace_back(std::move(ptr));
  }

  // Clear all shape info memory blocks
  void ClearShapeInfoPtrList() {
    shapeInfoPtrList.clear();
  }
  std::string BitsetToString(const std::bitset<MAX_SCOPE_NUM> &bitset) const;
  // Expand update nodes
  bool ExpandUpdateNodes(std::vector<SuperKernelBaseNode *> &customNodes);
  void SetUpdateFlag(bool flag) {
    needUpdate = flag;
  }
  // Parse original scopes from graph (before any splitting)
  // 3 cases:
  // 1. Has scopename non-empty scope nodes: group by scopeBitFlags
  // 2. No scope nodes at all: all nodes as one scope
  // 3. Only scopename empty scope nodes: all non-scope nodes as one scope
  void ParseOriginalScopes();

  // Get original scope infos (parsed before splitting)
  const std::vector<OriginalScopeInfo> &GetOriginalScopeInfos() const {
    return originalScopeInfos_;
  }

  // Dump all nodes' fusion fail reasons to log file
  void DumpFusionFailReasons(const std::vector<SuperKernelScopeInfo> &processedScopeInfos,
                             bool debugPerOpMaxCoreEnabled = false);

  // Convert graph to JSON format matching sk_raw_tasks_after.json structure
  Json ToJson() const;

  // 封装方法：从 modelRI 获取信息并初始化图（只获取流和任务信息，不执行后续处理）
  bool InitFromModelRI();

  // Fusion fail reasons statistics
  struct FusionFailStats {
    std::unordered_map<std::string, size_t> reasonStats;
    size_t fusibleCount = 0;
    size_t unfusibleCount = 0;
    std::vector<std::string> nodeLogEntries;           // All nodes (for plog)
    std::vector<std::string> unfusibleNodeLogEntries;  // Only unfusible nodes (for log file)
  };

 private:
  // Collect fusion fail reasons statistics for all nodes
  FusionFailStats CollectFusionFailStats();
  bool AddNode(std::unique_ptr<SuperKernelBaseNode> node);
  bool AddEventAssociateNotify(uint64_t eventId, SuperKernelBaseNode *node);
  bool AddEventAssociateWait(uint64_t eventId, SuperKernelBaseNode *node);
  bool AddEventAssociateReset(uint64_t eventId, SuperKernelBaseNode *node);
  bool AddMemoryAssociateWrite(uint64_t eventId, SuperKernelBaseNode *node);
  bool AddMemoryAssociateWait(uint64_t eventId, SuperKernelBaseNode *node);
  bool AddEventAssociate();
  void BuildEventNodeAssociations();
  uint32_t GetValueBreakerBypass() const;
  bool PostProcessMemoryNode();
  bool ProcessMemoryWriteNodes(const uint64_t eventId, const MemoryInfos &memoryInfo, const uint64_t memoryWaitValue,
                               const uint32_t waitFlag);
  bool UpdateNodeScopeBitFlags();
  bool InitStreamsFromModelRI(std::vector<uint32_t> &streamTaskNums);
  bool ProcessAllStreamsAndTasks(const std::vector<uint32_t> &streamTaskNums);
  bool ProcessSingleTask(aclmdlRITask &task, uint32_t streamIdx, uint32_t taskIdx, uint64_t &preNodeId);
  void RegisterFusibleScope(const std::unique_ptr<SuperKernelBaseNode> &node);
  void UpdateNodeRelations(uint64_t nodeId, uint32_t streamIdx, uint32_t taskIdx, uint64_t &preNodeId);
  std::unordered_map<uint64_t, std::unique_ptr<SuperKernelBaseNode>> graphMap;
  std::unordered_map<uint64_t, EventInfos> eventToNodes;
  std::unordered_map<uint64_t, MemoryInfos> memoryToNodes;
  std::vector<uint64_t> headNodes;
  std::vector<uint64_t> nodeSizeInStream;
  std::vector<aclrtStream> streams;
  aclmdlRI modelRI;
  std::string modelId;
  friend class SuperKernelOptimizer;
  friend class ScopeVerifyGraphBuilder;
  std::unordered_map<std::string, uint32_t> scopeNameToIdx;  ///< scopeName -> scopeIdx
  std::unordered_map<uint32_t, std::string> scopeIdxToName;  ///< scopeIdx -> scopeName (reverse mapping)
  std::vector<OriginalScopeInfo> originalScopeInfos_;        ///< Original scopes before any splitting
  std::vector<std::unique_ptr<uint8_t[]>>
      shapeInfoPtrList;  ///< profiling sk shape info memory, lifecycle follows graph
  std::unordered_set<SuperKernelBaseNode *>
      needUpdateNodes;  ///< event nodes that need to be marked with scope bit flags after scope processing
  bool needUpdate = false;
  const SuperKernelOptionsManager *opts_ = nullptr;
};

#endif  // __SK_GRAPH_H__
