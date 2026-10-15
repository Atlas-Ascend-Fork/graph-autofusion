/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include "sk_optimizer.h"
#include "sk_scope_split.h"
#include "sk_scope_postprocess.h"
#include "sk_task_builder.h"
#include "sk_log.h"
#include "sk_dump_json.h"
#include "aprof_pub.h"
#include "securec.h"
#include "sk_event_recorder.h"

namespace {
void PrintSKNodesDetail(std::string skFuncName, SuperKernelScopeInfo &scopeInfo) {
  uint16_t scopeId = scopeInfo.GetScopeId();
  auto &nodes = scopeInfo.GetExtInfo().filteredNodes;
  SK_LOGI("  SK Function: %s, scope id: %u, Node Count: %zu", skFuncName.c_str(), scopeId, nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    SK_LOGI("    [%zu] %s", i, nodes[i]->Format().c_str());
  }
}

void PrintSKNodesWithLaunchInfoDetail(std::string skFuncName, SuperKernelScopeInfo &scopeInfo,
                                      const SkLaunchInfo &launchInfo) {
  uint16_t scopeId = scopeInfo.GetScopeId();
  auto &nodes = scopeInfo.GetExtInfo().filteredNodes;
  SK_LOGI("  SK Function: %s, scope id: %u, Node Count: %zu, kernelType: %s, numBlocks: %u, batchMode: %d",
          skFuncName.c_str(), scopeId, nodes.size(), to_string(launchInfo.entryInfo.entryType),
          launchInfo.entryInfo.numBlocks, static_cast<int>(launchInfo.isScheModeOn));
  for (size_t i = 0; i < nodes.size(); ++i) {
    SK_LOGI("    [%zu] %s", i, nodes[i]->Format().c_str());
  }
}

void PrintSKNodes(std::string skFuncName, SuperKernelScopeInfo &scopeInfo, const SkLaunchInfo &launchInfo) {
  {
    SK_LOG_CONTEXT_SIMPLE("sk_fused_nodes.log");
    PrintSKNodesWithLaunchInfoDetail(skFuncName, scopeInfo, launchInfo);
  }
  PrintSKNodesDetail(skFuncName, scopeInfo);
}

void PrintTaskNodesDetail(const std::vector<SuperKernelBaseNode *> &nodes, const char *tag) {
  SK_LOGI("%s: node count=%zu", tag, nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    SuperKernelBaseNode *node = nodes[i];
    if (node == nullptr) {
      SK_LOGI("  [%zu] nullptr", i);
      continue;
    }
    SK_LOGI("  [%zu] %s", i, node->Format().c_str());
  }
}
}  // namespace

std::vector<SuperKernelBaseNode *> SuperKernelOptimizer::ReorderWaitNodesForTaskBuild(
    const std::vector<SuperKernelBaseNode *> &taskNodes) const {
  if (!ShouldReorderWaitNodesForTaskBuild()) {
    SK_LOGI("task reorder disabled: auto_op_parallel is not CUSTOMIZE_QUEUE");
    return taskNodes;
  }

  struct PendingWaitNode {
    SuperKernelBaseNode *node = nullptr;
    size_t originalIdx = 0;
    size_t targetKernelIdx = 0;
  };

  const size_t invalidKernelIdx = taskNodes.size();
  std::unordered_map<uint32_t, size_t> nextKernelIdxByStream;
  std::vector<size_t> waitTargetKernelIdx(taskNodes.size(), invalidKernelIdx);

  for (size_t idx = taskNodes.size(); idx > 0; --idx) {
    size_t curIdx = idx - 1;
    SuperKernelBaseNode *node = taskNodes[curIdx];

    uint32_t streamIdx = node->GetStreamIdxInGraph();
    if (node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      nextKernelIdxByStream[streamIdx] = curIdx;
    } else if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
      auto kernelIt = nextKernelIdxByStream.find(streamIdx);
      if (kernelIt != nextKernelIdxByStream.end()) {
        waitTargetKernelIdx[curIdx] = kernelIt->second;
      }
    }
  }

  size_t moveCount = 0;
  std::vector<PendingWaitNode> pendingWaitNodes;
  std::vector<SuperKernelBaseNode *> reorderedTaskNodes;
  reorderedTaskNodes.reserve(taskNodes.size());

  auto flushPendingWaitNodes = [&](size_t currentIdx) {
    std::vector<PendingWaitNode> remainedWaitNodes;
    remainedWaitNodes.reserve(pendingWaitNodes.size());
    bool hasReadyWaitNode = false;
    for (const auto &pendingWaitNode : pendingWaitNodes) {
      if (pendingWaitNode.targetKernelIdx > currentIdx) {
        remainedWaitNodes.push_back(pendingWaitNode);
        continue;
      }

      hasReadyWaitNode = true;
      size_t finalIdx = reorderedTaskNodes.size();
      reorderedTaskNodes.push_back(pendingWaitNode.node);
      if (finalIdx != pendingWaitNode.originalIdx) {
        ++moveCount;
      }
      SK_LOGI(
          "task reorder: place deferred wait node, waitNodeId=%lu, streamIdx=%u, originalIdx=%zu, finalIdx=%zu, "
          "targetKernelIdx=%zu",
          pendingWaitNode.node->GetNodeId(), pendingWaitNode.node->GetStreamIdxInGraph(), pendingWaitNode.originalIdx,
          finalIdx, pendingWaitNode.targetKernelIdx);
    }
    if (!hasReadyWaitNode) {
      return;
    }
    pendingWaitNodes.swap(remainedWaitNodes);
  };

  for (size_t idx = 0; idx < taskNodes.size(); ++idx) {
    SuperKernelBaseNode *node = taskNodes[idx];
    if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
      uint32_t streamIdx = node->GetStreamIdxInGraph();
      if (waitTargetKernelIdx[idx] == invalidKernelIdx) {
        SK_LOGI(
            "task reorder: keep wait node in place, waitNodeId=%lu, "
            "streamIdx=%u, originalIdx=%zu, reason=no later kernel in same stream",
            node->GetNodeId(), streamIdx, idx);
        flushPendingWaitNodes(idx);
        reorderedTaskNodes.push_back(node);
        continue;
      }

      SuperKernelBaseNode *targetKernelNode = taskNodes[waitTargetKernelIdx[idx]];
      SK_LOGI(
          "task reorder: defer wait node for same-stream kernel, waitNodeId=%lu, "
          "streamIdx=%u, originalIdx=%zu, targetKernelIdx=%zu, targetKernelNodeId=%lu",
          node->GetNodeId(), streamIdx, idx, waitTargetKernelIdx[idx],
          targetKernelNode == nullptr ? INVALID_TASK_ID : targetKernelNode->GetNodeId());
      pendingWaitNodes.push_back({node, idx, waitTargetKernelIdx[idx]});
      continue;
    }

    flushPendingWaitNodes(idx);
    reorderedTaskNodes.push_back(node);
  }

  flushPendingWaitNodes(taskNodes.size());
  SK_LOGI("task reorder finished: originalCount=%zu, moveCount=%zu", taskNodes.size(), moveCount);
  return reorderedTaskNodes;
}

bool SuperKernelOptimizer::ShouldReorderWaitNodesForTaskBuild() const {
  const auto *autoOpParallel = opts.GetOption(aclskOptionType::AUTO_OP_PARALLEL);
  if (autoOpParallel == nullptr) {
    return false;
  }
  return static_cast<SkHeapType>(autoOpParallel->GetIntValue()) == SkHeapType::CUSTOMIZE_QUEUE;
}

bool SuperKernelOptimizer::Update(SuperKernelScopeInfo &scopeInfo, SuperKernelGraph &graph,
                                  const SkLaunchInfo &launchInfo) {
  const std::string &scopeName = scopeInfo.GetExtInfo().scopeName;
  SK_LOGI("scope update begin: scopeName=%s, streamCount=%zu", scopeName.c_str(),
          scopeInfo.GetScopeStreamInfos().size());
  bool skMainNodeUpdated = false;
  size_t updateTotalCount = 0;

  auto &scopeStreamInfos = scopeInfo.GetScopeStreamInfos();
  auto &extInfo = scopeInfo.MutableExtInfo();
  for (size_t streamIdx = 0; streamIdx < scopeStreamInfos.size(); ++streamIdx) {
    auto &streamInfo = scopeStreamInfos[streamIdx];
    auto &customParams = extInfo.customParamsList[streamIdx];
    SK_LOGI(
        "update stream begin: scopeName=%s, streamId=%u, headNodeId=%lu, tailNodeId=%lu, nodeSize=%lu, "
        "customParamSize=%zu",
        scopeName.c_str(), streamInfo.streamIdx, streamInfo.headNodeIdx, streamInfo.tailNodeIdx, streamInfo.nodeSize,
        customParams.size());
    size_t customParamSize = customParams.size();
    if (streamInfo.nodeSize < customParamSize) {
      SK_LOGE("node size is less than custom params size: nodeSize=%lu, customParamSize=%zu", streamInfo.nodeSize,
              customParamSize);
      return false;
    }

    uint64_t curNodeId = streamInfo.headNodeIdx;
    size_t eventCnt = 0;

    while (curNodeId != INVALID_TASK_ID) {
      auto *node = graph.GetNodeById(curNodeId);
      if (node == nullptr) {
        SK_LOGE("node not found during stream-based update: nodeId=%lu, streamId=%u", curNodeId, streamInfo.streamIdx);
        return false;
      }
      UpdateContext ctx;
      if (eventCnt < customParamSize) {
        auto &curCustomParams = customParams[eventCnt++];
        ctx.customParams = &curCustomParams;
      } else if (curNodeId == extInfo.skMainNodeId) {
        if (!skMainNodeUpdated) {
          skMainNodeUpdated = true;
          ctx.launchInfo = const_cast<SkLaunchInfo *>(&launchInfo);
        } else {
          SK_LOGI("repeat find sk launch node, skip update kernel and set invalid node");
        }
      }

      ++updateTotalCount;
      if (!node->Update(ctx)) {
        SK_LOGE("node update failed: nodeId=%lu, streamId=%u", curNodeId, streamInfo.streamIdx);
        return false;
      }

      if (curNodeId == streamInfo.tailNodeIdx) {
        break;
      }
      curNodeId = node->GetNextNodeId();
    }
    SK_LOGI("update stream end: scopeName=%s, streamId=%u, visitedNodes=%zu", scopeName.c_str(), streamInfo.streamIdx,
            eventCnt);
  }

  if (!skMainNodeUpdated) {
    SK_LOGE("failed to find sk launch node, sk optimize failed");
    return false;
  }

  graph.SetUpdateFlag(true);
  SK_LOGI("scope update finished: scopeName=%s, updateTotalNodes=%zu", scopeName.c_str(), updateTotalCount);
  return true;
}

// Schedule task-stream nodes for super-kernel launch.
bool SuperKernelOptimizer::Schedule(SuperKernelScopeInfo &scopeInfo, SuperKernelGraph &graph, SkTaskBuilder &builder) {
  const auto &taskNodes = scopeInfo.GetExtInfo().filteredNodes;
  if (taskNodes.empty()) {
    SK_LOGE("no tasks for super kernel optimization: scope has 0 nodes for optimization");
    return false;
  }
  std::vector<SuperKernelBaseNode *> reorderedTaskNodes = ReorderWaitNodesForTaskBuild(taskNodes);
  if (reorderedTaskNodes.size() != taskNodes.size()) {
    SK_LOGE("task reorder produced invalid size: originalCount=%zu, reorderedCount=%zu", taskNodes.size(),
            reorderedTaskNodes.size());
    return false;
  }

  std::string skFuncName = GetSkFuncName(reorderedTaskNodes, scopeInfo.GetScopeId(), scopeInfo.GetExtInfo().scopeName);
  PrintTaskNodesDetail(reorderedTaskNodes, "reordered task nodes");

  std::vector<SuperKernelBaseNode *> customTasks;
  customTasks.reserve(scopeInfo.GetExtInfo().eventNodes.size());
  for (const auto &eventNode : scopeInfo.GetExtInfo().eventNodes) {
    customTasks.emplace_back(eventNode.get());
  }

  SK_LOGI("schedule scope: taskCount=%zu, customTaskCount=%zu, updateStreamCount=%zu", reorderedTaskNodes.size(),
          customTasks.size(), scopeInfo.GetScopeStreamInfos().size());

  SkBuildResult buildResult =
      builder.Build(skFuncName, reorderedTaskNodes, customTasks, scopeInfo.GetScopeId(), scopeInfo.GetScopeCoreInfo());
  SkLaunchInfo &launchInfo = buildResult.launchInfo;

  // Collect task queue JSON for this scope
  taskQueueJsons_[std::to_string(scopeInfo.GetScopeId())] = buildResult.taskQueueJson;

  if (!SkProfiling(scopeInfo, launchInfo, graph)) {
    SK_LOGE("SkProfiling failed");
    return false;
  }
  if (!DumpProfilingDetail(reorderedTaskNodes, launchInfo, scopeInfo, graph)) {
    SK_LOGE("Dump sk time profiling detail failed");
    return false;
  }

  if (launchInfo.entryInfo.skEntryFunc == nullptr || launchInfo.devArgs.Get() == nullptr) {
    SK_LOGE("schedule failed: build launch info failed");
    return false;
  }
  PrintSKNodes(skFuncName, scopeInfo, launchInfo);
  SK_LOGI(
      "schedule scope: build finished, entryType=%s, entryFuncHandle=%p, skFuncName=%s, useSimtEntry=%d, "
      "skMaxDcacheSize=%zu, isScheModeOn=%d",
      to_string(launchInfo.entryInfo.entryType), launchInfo.entryInfo.skEntryFunc, launchInfo.skFuncName.c_str(),
      static_cast<int>(launchInfo.useSimtEntry), launchInfo.skMaxDcacheSize, launchInfo.isScheModeOn);

  if (!Update(scopeInfo, graph, launchInfo)) {
    SK_LOGE("schedule failed: scope update failed");
    return false;
  }
  return true;
}

bool SuperKernelOptimizer::Process(SuperKernelGraph &graph) {
  SuperKernelScopeSplitter splitter(graph, opts);
  if (splitter.SplitGraph()) {
    SK_LOGI("graph split into %zu scopes", splitter.GetScopeInfos().size());
  } else {
    SK_LOGE("graph split failed or no scopes found: cannot proceed with super kernel optimization");
    return false;
  }
  // Save scope infos for later access (e.g., dump to JSON)
  processedScopeInfos_ = std::move(splitter.GetScopeInfos());

  SkTaskBuilder builder(opts, graph);
  SuperKernelScopePostProcessor postProcessor(graph);

  for (size_t i = 0; i < processedScopeInfos_.size(); ++i) {
    auto &scopeInfo = processedScopeInfos_[i];
    SK_LOGI("process scope begin: scopeId=%u", scopeInfo.GetScopeId());
    if (!postProcessor.PostProcess(scopeInfo)) {
      const ScopeProcessStatus processStatus = scopeInfo.GetExtInfo().processStatus;
      if (processStatus == ScopeProcessStatus::UNRECOVERABLE_FAIL) {
        SK_LOGE(
            "scope processing encountered an unrecoverable failure that cannot be skipped, "
            "abort super kernel optimization: scopeId=%u, scopeStatus=%s",
            scopeInfo.GetScopeId(), to_string(processStatus));
        return false;
      }
      SK_LOGI("scope unprocessable, skip schedule/update: scopeId=%u, scopeStatus=%s", scopeInfo.GetScopeId(),
              to_string(processStatus));
      // Set fusion fail reason for all nodes in this scope with scope detail
      for (auto *node : scopeInfo.GetNodes()) {
        if (node != nullptr) {
          node->SetFusionFailReason(processStatus);
          node->SetIsFusible(false);
        }
      }
      continue;
    }
    if (scopeInfo.GetExtInfo().filteredNodes.empty()) {
      SK_LOGI("scope has no nodes after processing, skipping schedule/update: scopeId=%u", scopeInfo.GetScopeId());
      continue;
    }

    scopeInfo.MutableExtInfo().scopeName =
        ScopeSplitPass::GetScopeNamesFromBitFlags(scopeInfo.GetScopeBitFlags(), graph);
    if (!Schedule(scopeInfo, graph, builder)) {
      SK_LOGE("process scope failed: scopeId=%u, schedule/update returned false", scopeInfo.GetScopeId());
      return false;
    }
  }

  // Dump all nodes' fusion fail reasons after all scopes are processed
  graph.DumpFusionFailReasons(processedScopeInfos_, splitter.IsDebugPerOpMaxCoreEnabled());

  // Dump all task queues to a single JSON file
  if (!taskQueueJsons_.empty()) {
    SK_LOGI("Dumping all task queues to JSON, scopeCount=%zu", taskQueueJsons_.size());
    if (!DumpAllTaskQueuesToJson(graph, taskQueueJsons_)) {
      SK_LOGE("Failed to dump all task queues to JSON, continuing...");
    }
  }

  SK_LOGI("super kernel process finished: scopeCount=%zu", processedScopeInfos_.size());
  return true;
}
