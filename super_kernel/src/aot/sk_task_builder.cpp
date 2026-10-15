/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sk_task_builder.h"
#include "sk_graph.h"
#include "sk_dump_json.h"
#include "sk_log.h"
#include "sk_common.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include "securec.h"
#include "sk_event_recorder.h"
#include "runtime/rt_external_kernel.h"
#include "runtime/kernel.h"

extern "C" aclrtBinHandle AscendGetEntryBinHandle();

namespace {
constexpr size_t kCounterAlignBytes = 64;
constexpr uint64_t kDebugOptionDisableDcci = 0x1;
constexpr uint64_t kDebugOptionDebugSyncAll = 0x2;
constexpr uint64_t kDebugOptionDcciBeforeKernelStart = 0x4;
constexpr uint64_t kDebugOptionDcciAfterKernelEnd = 0x8;
constexpr uint64_t kDebugOptionCrossCoreSyncCheck = 0x10;
constexpr uint64_t kDebugOptionEnableDcciAfterFunc = 0x20;

inline size_t AlignUp(size_t value, size_t align) {
  if (align == 0) {
    return value;
  }
  return ((value + align - 1) / align) * align;
}

SkQueueType ToQueueType(SkKernelType kernelType) {
  switch (kernelType) {
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
      SK_LOGW("Unsupported kernel type %s for super kernel; using AIC as fallback", to_string(kernelType));
      return SkQueueType::AIC;
  }
}

aclrtFuncHandle ResolveSkEntryFunc(const char *funcName) {
  aclrtBinHandle bhdl = AscendGetEntryBinHandle();
  if (bhdl == nullptr) {
    SK_LOGE("failed to get entry bin handle: AscendGetEntryBinHandle() returned null");
    return nullptr;
  }
  aclrtFuncHandle fhdl = nullptr;
  CHECK_ACL(aclrtBinaryGetFunction(bhdl, funcName, &fhdl));
  if (fhdl == nullptr) {
    SK_LOGE("failed to resolve entry func handle: funcName=%s, binHandle=%p", funcName, bhdl);
    return nullptr;
  }
  return fhdl;
}

bool UsesAic(SkQueueType type) {
  return type == SkQueueType::AIC || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

bool UsesAiv(SkQueueType type) {
  return type == SkQueueType::AIV || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

const KernelInfos &GetKernelInfos(const SuperKernelBaseNode *node) {
  return node->GetNodeInfos().kernelInfos;
}

const SkKernelType GetKernelType(const SuperKernelBaseNode *node) {
  if (node == nullptr) {
    return SkKernelType::DEFAULT;
  }
  auto nodeType = node->GetNodeType();
  if (nodeType == SkNodeType::NODE_KERNEL) {
    return GetKernelInfos(node).kernelType;
  }
  return SkKernelType::DEFAULT;
}

bool ValidateTaskCoreRequirement(const SuperKernelBaseNode &node, const ScopeCoreInfo &scopeCoreInfo) {
  const uint32_t skCubeNum = scopeCoreInfo.GetCubeNum();
  const uint32_t skVectorNum = scopeCoreInfo.GetVectorNum();
  const uint32_t cubeNum = node.GetCubeNum();
  const uint32_t vectorNum = node.GetVecNum();
  if (node.GetKernelType() == SkKernelType::MIX_AIC_1_2 && scopeCoreInfo.type != SkKernelType::MIX_AIC_1_2) {
    return false;
  }
  if (node.RequiresExactCoreMatch()) {
    if (cubeNum > skCubeNum || vectorNum > skVectorNum) {
      return false;
    }
    // ScheMode requires an exact match: pure-vector kernels match SK vector cores; others match SK cube cores.
    if (cubeNum == 0 && vectorNum != 0) {
      return vectorNum == skVectorNum;
    } else {
      return cubeNum == skCubeNum;
    }
  } else {
    // blockDimScaleUp adapts at dispatch time and follows the non-ScheMode upper-bound check.
    return cubeNum <= skCubeNum && vectorNum <= skVectorNum;
  }
}

uint32_t GetTaskNumBlocks(const SuperKernelBaseNode &node, SkQueueType queueType, const ScopeCoreInfo &scopeCoreInfo) {
  const bool useVector = queueType == SkQueueType::AIV;
  if (node.GetNodeInfos().kernelInfos.capBits.blockDimScaleUp) {
    return useVector ? scopeCoreInfo.GetVectorNum() : scopeCoreInfo.GetCubeNum();
  }

  const SkKernelType kernelType = node.GetKernelType();
  // MIX tasks in a MIX1:2 SK use two vector blocks per cube block.
  const bool adaptToMix12 = useVector && scopeCoreInfo.type == SkKernelType::MIX_AIC_1_2 &&
                            (kernelType == SkKernelType::MIX_AIC_1_1 || kernelType == SkKernelType::MIX_AIC_1_2);
  return adaptToMix12 ? node.GetNumBlocks() * 2U : node.GetNumBlocks();
}

SkQueueType ToEventQueueType(SkQueueType queueType, bool aicAvailable = true, bool aivAvailable = true) {
  SkQueueType eventQueueType = (queueType == SkQueueType::AIC) ? SkQueueType::AIC : SkQueueType::AIV;
  if (eventQueueType == SkQueueType::AIC && !aicAvailable) {
    return SkQueueType::AIV;
  }
  if (eventQueueType == SkQueueType::AIV && !aivAvailable) {
    return SkQueueType::AIC;
  }
  return eventQueueType;
}

SkQueueType InferFirstKernelEventQueueType(const std::vector<SuperKernelBaseNode *> &tasks, bool aicAvailable = true,
                                           bool aivAvailable = true) {
  for (const auto *node : tasks) {
    if (node != nullptr && node->GetNodeType() == SkNodeType::NODE_KERNEL) {
      return ToEventQueueType(ToQueueType(GetKernelInfos(node).kernelType), aicAvailable, aivAvailable);
    }
  }
  return SkQueueType::UNKNOWN;
}

struct SimtDcacheSizeResult {
  bool isValid = true;
  bool hasSimtTask = false;
  size_t skMaxDcacheSize = 0;
};

SimtDcacheSizeResult CalculateSimtDcacheSize(const std::vector<SuperKernelBaseNode *> &tasks) {
  SimtDcacheSizeResult dcacheSizeResult;
  size_t skMaxDcacheSize = SK_TOTAL_UB_SIZE;
  for (const auto *task : tasks) {
    if (task == nullptr || task->GetNodeType() != SkNodeType::NODE_KERNEL) {
      continue;
    }
    const KernelInfos &kernelInfo = GetKernelInfos(task);
    if (!kernelInfo.isSimtOp) {
      continue;
    }
    dcacheSizeResult.hasSimtTask = true;
    if (!kernelInfo.hasDynUbufSize || !kernelInfo.hasAllocUbufSize) {
      SK_LOGE("SIMT kernel lacks ubuf size info, nodeId=%lu, hasDynUbufSize=%d, hasAllocUbufSize=%d", task->GetNodeId(),
              kernelInfo.hasDynUbufSize, kernelInfo.hasAllocUbufSize);
      dcacheSizeResult.isValid = false;
      return dcacheSizeResult;
    }
    if (kernelInfo.dynUbufSize > SK_TOTAL_UB_SIZE ||
        kernelInfo.allocUbufSize > SK_TOTAL_UB_SIZE - kernelInfo.dynUbufSize) {
      SK_LOGE(
          "SIMT kernel ubuf size exceeds total ub size, nodeId=%lu, dynUbufSize=%zu, "
          "allocUbufSize=%zu, totalUbSize=%zu",
          task->GetNodeId(), kernelInfo.dynUbufSize, kernelInfo.allocUbufSize, SK_TOTAL_UB_SIZE);
      dcacheSizeResult.isValid = false;
      return dcacheSizeResult;
    }
    size_t taskMaxDcacheSize = SK_TOTAL_UB_SIZE - kernelInfo.dynUbufSize - kernelInfo.allocUbufSize;
    skMaxDcacheSize = std::min(skMaxDcacheSize, taskMaxDcacheSize);
  }
  if (dcacheSizeResult.hasSimtTask) {
    dcacheSizeResult.skMaxDcacheSize = skMaxDcacheSize;
  }
  return dcacheSizeResult;
}

bool MatchKernelOption(const SuperKernelOptionsManager &opts, const std::vector<std::string> &kernelList,
                       const std::string &funcName) {
  return !kernelList.empty() && !funcName.empty() && opts.MatchKernelNameInList(kernelList, funcName);
}

uint64_t BuildFuncTaskDebugOptions(const SuperKernelOptionsManager &opts, const KernelInfos &kernelInfo,
                                   const std::vector<std::string> &disableDcciList,
                                   const std::vector<std::string> &dcciBeforeKernelStartList,
                                   const std::vector<std::string> &dcciAfterKernelEndList,
                                   uint32_t debugCrossCoreSyncCheck) {
  const bool disableDcciByCap = kernelInfo.capBits.disableDcci;
  const bool disableDcciByOption = MatchKernelOption(opts, disableDcciList, kernelInfo.funcName);
  const bool dcciBeforeKernelStart = MatchKernelOption(opts, dcciBeforeKernelStartList, kernelInfo.funcName);
  bool dcciAfterKernelEnd = MatchKernelOption(opts, dcciAfterKernelEndList, kernelInfo.funcName);

  if (disableDcciByOption && !disableDcciByCap) {
    SK_LOGW(
        "DCCI disable option and SK_BIND mismatch for %s: dcci_disable_on_kernel=true, "
        "but SK_BIND disable_dcci=0. The sub-kernel binary may be reused from stale cache.",
        kernelInfo.funcName.c_str());
  }

  if (!disableDcciByCap && disableDcciByOption && !dcciBeforeKernelStart && !dcciAfterKernelEnd) {
    dcciAfterKernelEnd = true;
    SK_LOGI(
        "Apply dcci_after_kernel_end fallback for %s: SK_BIND disable_dcci=0, "
        "dcci_disable_on_kernel option matched, before/after dcci options not matched",
        kernelInfo.funcName.c_str());
  }

  uint64_t debugOptions = 0;
  if (disableDcciByCap || disableDcciByOption) {
    debugOptions |= kDebugOptionDisableDcci;
  }
  if (dcciBeforeKernelStart) {
    debugOptions |= kDebugOptionDcciBeforeKernelStart;
  }
  if (dcciAfterKernelEnd) {
    debugOptions |= kDebugOptionDcciAfterKernelEnd;
  }
  if (debugCrossCoreSyncCheck == 1) {
    debugOptions |= kDebugOptionCrossCoreSyncCheck;
  }
  if ((debugOptions & kDebugOptionDcciAfterKernelEnd) != 0 || (debugOptions & kDebugOptionDisableDcci) == 0) {
    debugOptions |= kDebugOptionEnableDcciAfterFunc;
  }
  return debugOptions;
}

SkQueueType ResolveMixWaitEventQueueType(const SuperKernelBaseNode *prevKernel, const SuperKernelBaseNode *mixKernel) {
  if (prevKernel == nullptr) {
    return SkQueueType::AIV;
  }

  SkQueueType prevKernelQueueType =
      (prevKernel == nullptr) ? SkQueueType::UNKNOWN : ToQueueType(GetKernelInfos(prevKernel).kernelType);

  if (prevKernelQueueType != SkQueueType::AIC && prevKernelQueueType != SkQueueType::AIV) {
    return SkQueueType::AIV;
  }

  const bool sameStream = prevKernel->GetStreamIdxInGraph() == mixKernel->GetStreamIdxInGraph();
  const bool hasDirectDep = prevKernel->sendToNodeId.count(mixKernel->GetNodeId()) > 0 ||
                            mixKernel->receiveNodeId.count(prevKernel->GetNodeId()) > 0;
  return (sameStream || hasDirectDep)
             ? prevKernelQueueType
             : ((prevKernelQueueType == SkQueueType::AIV) ? SkQueueType::AIC : SkQueueType::AIV);
}

uint64_t GetTaskNodeIdByIndex(const std::vector<SuperKernelBaseNode *> &tasks, uint32_t taskIndex) {
  return taskIndex < tasks.size() ? tasks[taskIndex]->GetNodeId() : INVALID_TASK_ID;
}

// dump device entry args
void DumpTaskQueDetail(const TaskQue *que, const char *name, const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGD("%s TaskQue: cap=%u, tasks=%u", name, que->cap, que->taskCnt);
  for (uint32_t i = 0; i < que->taskCnt; ++i) {
    const TaskInfo &ti = que->taskInfos[i];
    const uint64_t nodeId = GetTaskNodeIdByIndex(tasks, ti.index);
    SK_LOGD(
        "[%u] type=%s, idx=%u, nodeId=%lu, relatedType=%s, blk=%u, entries=%u, args=0x%llx, argsSize=%u, "
        "debugOptions=0x%llx, extraInfo=0x%llx",
        i, to_string(ti.type), ti.index, nodeId, to_string(ti.relatedType), ti.numBlocks, ti.entryCnt,
        (unsigned long long)ti.args, ti.argsSize, (unsigned long long)ti.debugOptions,
        (unsigned long long)ti.extraInfo);
    const bool isEventTask = ti.type == SkTaskType::TYPE_EVENT_NOTIFY || ti.type == SkTaskType::TYPE_EVENT_WAIT ||
                             ti.type == SkTaskType::TYPE_EVENT_RESET;
    if (isEventTask) {
      SK_LOGD("   eventValue=0x%llx", (unsigned long long)GetEventTaskValue(ti));
    }
    if (ti.type == SkTaskType::TYPE_EVENT_WAIT) {
      const auto waitFlag = static_cast<SkMemoryWaitFlag>(GetEventTaskFlag(ti));
      SK_LOGD("   waitFlag=%s(%u)", to_string(waitFlag), static_cast<uint32_t>(waitFlag));
    }
    for (uint32_t j = 0; j < ti.entryCnt; ++j) {
      SK_LOGD("   entry[%u]=0x%llx", j, (unsigned long long)ti.entry[j]);
    }
  }
}

void DumpDeviceArgsDetail(const std::string &skFuncName, const SkDeviceEntryArgs *args,
                          const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGD("Dumping device args for function: %s", skFuncName.c_str());
  const uint8_t *base = (const uint8_t *)args;
  const SkHeaderInfo &hdr = args->skHeader;
  SK_LOGD(
      "SkHeaderInfo: aicOff=%u, aivOff=%u, counterOff=%u, dfxOff=%u, eventConfigOff=%u, nodeCnt=%u, totalSize=%lu\n",
      hdr.aicQueOffset, hdr.aivQueOffset, hdr.counterOffset, hdr.dfxOffset, hdr.eventConfigOffset, hdr.nodeCnt,
      hdr.totalSize);

  DumpTaskQueDetail((const TaskQue *)(base + args->skHeader.aicQueOffset), "AIC", tasks);
  DumpTaskQueDetail((const TaskQue *)(base + args->skHeader.aivQueOffset), "AIV", tasks);

  const SkDfxInfo *dfx = (const SkDfxInfo *)(base + args->skHeader.dfxOffset);
  for (uint32_t i = 0; i < args->skHeader.nodeCnt; ++i) {
    SK_LOGD("dfx[%u]: bin=0x%llx, ori=0x%llx, aicSize=0x%x, aivSize=0x%x, numBlocks=%u, cubeNum=%u, vecNum=%u", i,
            (unsigned long long)dfx[i].binHdl, (unsigned long long)dfx[i].funcHdlOri, dfx[i].aicSize, dfx[i].aivSize,
            dfx[i].numBlocks, dfx[i].cubeNum, dfx[i].vecNum);
    SK_LOGD("  entryAic[0]=0x%llx, entryAic[1]=0x%llx, entryAic[2]=0x%llx, entryAic[3]=0x%llx",
            (unsigned long long)dfx[i].entryAic[0], (unsigned long long)dfx[i].entryAic[1],
            (unsigned long long)dfx[i].entryAic[2], (unsigned long long)dfx[i].entryAic[3]);
    SK_LOGD("  entryAiv[0]=0x%llx, entryAiv[1]=0x%llx, entryAiv[2]=0x%llx, entryAiv[3]=0x%llx",
            (unsigned long long)dfx[i].entryAiv[0], (unsigned long long)dfx[i].entryAiv[1],
            (unsigned long long)dfx[i].entryAiv[2], (unsigned long long)dfx[i].entryAiv[3]);
  }
}

void DumpDeviceArgs(std::string skFuncName, const SkDeviceEntryArgs *args,
                    const std::vector<SuperKernelBaseNode *> &tasks) {
  {
    SK_LOG_CONTEXT_SIMPLE("sk_device_args.log");
    DumpDeviceArgsDetail(skFuncName, args, tasks);
  }
  DumpDeviceArgsDetail(skFuncName, args, tasks);
}

// ========== SkTaskBuilder static helper implementations ==========

// Search direction enum
enum class SearchDirection : uint8_t { PREV, NEXT };
const char *to_string(SearchDirection dir) {
  return (dir == SearchDirection::PREV) ? "PREV" : "NEXT";
}

std::string BuildSearchPathString(const std::vector<uint64_t> &path) {
  std::string pathStr;
  for (size_t idx = 0; idx < path.size(); ++idx) {
    if (idx > 0) {
      pathStr += "->";
    }
    pathStr += std::to_string(path[idx]);
  }
  return pathStr.empty() ? "empty" : pathStr;
}

void LogKernelSearchFailure(uint64_t startNodeId, SearchDirection direction, const std::vector<uint64_t> &path,
                            const char *reason, uint64_t failedNodeId) {
  std::string pathStr = BuildSearchPathString(path);
  SK_LOGI("FindKernelNodeInDirection unsuccessful: startNodeId=%lu, direction=%s, reason=%s, failedNodeId=%lu, path=%s",
          startNodeId, to_string(direction), reason, failedNodeId, pathStr.c_str());
}

const SuperKernelBaseNode *ResolveCurrentKernel(
    uint64_t curNodeId, const SuperKernelBaseNode *current,
    const std::unordered_map<uint64_t, const SuperKernelBaseNode *> &cache) {
  auto it = cache.find(curNodeId);
  if (it != cache.end()) {
    return it->second;
  }
  if (current->GetNodeType() == SkNodeType::NODE_KERNEL) {
    return current;
  }
  return nullptr;
}

bool CacheTraversedNodes(const SuperKernelGraph &graph, const std::vector<uint64_t> &path, SkNodeType startNodeType,
                         const SuperKernelBaseNode *result,
                         std::unordered_map<uint64_t, const SuperKernelBaseNode *> &cache, uint64_t &failedNodeId) {
  // This cache is intentionally direction-agnostic: we only care whether a node
  // has already been attached to a reusable kernel for later inference.
  for (uint64_t nodeId : path) {
    auto *pathNode = graph.GetNodeById(nodeId);
    if (pathNode == nullptr) {
      failedNodeId = nodeId;
      return false;
    }
    auto nodeType = pathNode->GetNodeType();
    if (nodeType == SkNodeType::NODE_KERNEL || nodeType == startNodeType) {
      cache[nodeId] = result;
    }
  }
  return true;
}

// Find a KERNEL node in the given direction, return nullptr when not found.
const SuperKernelBaseNode *FindKernelNodeInDirection(uint64_t startNodeId, const SuperKernelGraph &graph,
                                                     SearchDirection direction,
                                                     std::unordered_map<uint64_t, const SuperKernelBaseNode *> &cache,
                                                     int maxHops = 100) {
  if (startNodeId == INVALID_TASK_ID) {
    SK_LOGI("FindKernelNodeInDirection skipped: startNodeId is INVALID_TASK_ID, direction=%s", to_string(direction));
    return nullptr;
  }

  const auto *startNode = graph.GetNodeById(startNodeId);
  if (startNode == nullptr) {
    SK_LOGI("FindKernelNodeInDirection incomplete: startNodeId=%lu, direction=%s, reason=start-node-not-found",
            startNodeId, to_string(direction));
    return nullptr;
  }
  const SkNodeType startNodeType = startNode->GetNodeType();

  uint64_t curNodeId = startNodeId;
  std::vector<uint64_t> path;  // Track all visited node IDs along traversal path.
  for (int i = 0; i < maxHops; ++i) {
    path.push_back(curNodeId);

    const auto *current = graph.GetNodeById(curNodeId);
    if (current == nullptr) {
      LogKernelSearchFailure(startNodeId, direction, path, "node-not-found", curNodeId);
      SK_LOGI("Node with ID %lu not found in graph.", curNodeId);
      return nullptr;
    }

    const auto *result = ResolveCurrentKernel(curNodeId, current, cache);
    if (result) {
      uint64_t failedNodeId = INVALID_TASK_ID;
      if (!CacheTraversedNodes(graph, path, startNodeType, result, cache, failedNodeId)) {
        LogKernelSearchFailure(startNodeId, direction, path, "path-node-not-found-during-cache-fill", failedNodeId);
        return nullptr;
      }
      return result;
    }

    curNodeId = (direction == SearchDirection::PREV) ? current->GetPreNodeId() : current->GetNextNodeId();
    if (curNodeId == INVALID_TASK_ID) {
      const char *reason = (direction == SearchDirection::PREV) ? "no-suitable-prev-node" : "no-suitable-next-node";
      LogKernelSearchFailure(startNodeId, direction, path, reason, path.back());
      SK_LOGI("nodeId:%lu has no %s-node.", startNodeId, to_string(direction));
      return nullptr;
    }

    if (std::find(path.cbegin(), path.cend(), curNodeId) != path.cend()) {
      LogKernelSearchFailure(startNodeId, direction, path, "loop-detected", curNodeId);
      SK_LOGI("nodeId:%lu detected loop in %s direction.", startNodeId, to_string(direction));
      return nullptr;
    }
  }

  LogKernelSearchFailure(startNodeId, direction, path, "max-hops-exceeded", curNodeId);
  SK_LOGI("nodeId:%lu search exceeded max hops (%d) in %s direction.", startNodeId, maxHops, to_string(direction));
  return nullptr;
}

SyncDirection GenSyncDirection(SkQueueType preType, SkQueueType currType) {
  // Equivalent to Python gen_sync_name.
  switch (preType) {
    case SkQueueType::MIX_1_1:
    case SkQueueType::MIX_1_2:
      if (currType == SkQueueType::MIX_1_1 || currType == SkQueueType::MIX_1_2) {
        return SyncDirection::MIX_TO_MIX;
      }
      return (currType == SkQueueType::AIC) ? SyncDirection::VEC_TO_CUB : SyncDirection::CUB_TO_VEC;
    case SkQueueType::AIC:
      switch (currType) {
        case SkQueueType::MIX_1_1:
        case SkQueueType::MIX_1_2:
          return SyncDirection::CUB_TO_VEC;
        case SkQueueType::AIC:
          return SyncDirection::CUB_TO_CUB;
        default:
          return SyncDirection::CUB_TO_VEC;
      }
    case SkQueueType::AIV:
      switch (currType) {
        case SkQueueType::MIX_1_1:
        case SkQueueType::MIX_1_2:
          return SyncDirection::VEC_TO_CUB;
        case SkQueueType::AIV:
          return SyncDirection::VEC_TO_VEC;
        default:
          return SyncDirection::VEC_TO_CUB;
      }
    default:
      return SyncDirection::VEC_TO_CUB;
  }
}

std::pair<bool, bool> CheckResourceStatus(const std::vector<SuperKernelBaseNode *> &tasks) {
  bool aicAvailable = false;
  bool aivAvailable = false;
  for (const auto *task : tasks) {
    if (task->GetNodeType() != SkNodeType::NODE_KERNEL) {
      continue;
    }
    const KernelInfos &kernelInfo = GetKernelInfos(task);
    aicAvailable |= UsesAic(ToQueueType(kernelInfo.kernelType));
    aivAvailable |= UsesAiv(ToQueueType(kernelInfo.kernelType));
    if (aicAvailable && aivAvailable) {
      break;
    }
  }
  return {aicAvailable, aivAvailable};
}

}  // namespace

// ========== Initialize sync metadata ==========
bool SkTaskBuilder::InitTaskSyncInfos(const std::vector<SuperKernelBaseNode *> &tasks) {
  taskSyncInfos_.clear();
  taskSyncInfos_.resize(tasks.size());

  std::tie(aicAvailable_, aivAvailable_) = CheckResourceStatus(tasks);
  if (!aicAvailable_ && !aivAvailable_) {
    SK_LOGE("no AIC or AIV resource detected in the tasks, unable to assign queue types.");
    return false;
  }
  // Lookup cache for nearest kernel nodes: nodeId -> kernelNode*
  std::unordered_map<uint64_t, const SuperKernelBaseNode *> kernelNodeCache;

  size_t kernelCount = 0;
  size_t notifyCount = 0;
  size_t waitCount = 0;
  size_t resetCount = 0;
  SkQueueType firstKernelEventQueueType = InferFirstKernelEventQueueType(tasks, aicAvailable_, aivAvailable_);
  SuperKernelBaseNode *prevKernelTask = nullptr;
  for (size_t i = 0; i < tasks.size(); i++) {
    SkNodeType nodeType = tasks[i]->GetNodeType();
    switch (nodeType) {
      case SkNodeType::NODE_KERNEL: {
        // KERNEL node: assign queueType from kernel type.
        const KernelInfos &kernelInfo = GetKernelInfos(tasks[i]);
        taskSyncInfos_[i].queueType = ToQueueType(kernelInfo.kernelType);
        prevKernelTask = tasks[i];
        kernelCount++;
        break;
      }
      case SkNodeType::NODE_NOTIFY: {
        // NOTIFY node: inherit type from previous KERNEL node
        auto *kernel =
            FindKernelNodeInDirection(tasks[i]->GetPreNodeId(), graph_, SearchDirection::PREV, kernelNodeCache);
        if (kernel == nullptr) {
          if (firstKernelEventQueueType == SkQueueType::UNKNOWN) {
            SK_LOGE("%s node %zu failed to resolve previous KERNEL node and fallback queue type is UNKNOWN, nodeId=%lu",
                    to_string(nodeType), i, tasks[i]->GetNodeId());
            return false;
          }
          taskSyncInfos_[i].queueType = firstKernelEventQueueType;
          SK_LOGI("%s node %zu unable to resolve previous KERNEL node, nodeId=%lu, fallback = %s", to_string(nodeType),
                  i, tasks[i]->GetNodeId(), to_string(taskSyncInfos_[i].queueType));
        } else {
          kernelNodeCache[tasks[i]->GetNodeId()] = kernel;  // cache current NOTIFY node for future searches
          // Event nodes are executed on a single queue selected by nearest kernel type.
          taskSyncInfos_[i].queueType =
              ToEventQueueType(ToQueueType(GetKernelInfos(kernel).kernelType), aicAvailable_, aivAvailable_);
        }
        notifyCount++;
        break;
      }
      case SkNodeType::NODE_RESET: {
        // RESET node is an event write-side task; inherit type from previous KERNEL node.
        auto *kernel =
            FindKernelNodeInDirection(tasks[i]->GetPreNodeId(), graph_, SearchDirection::PREV, kernelNodeCache);
        if (kernel == nullptr) {
          if (firstKernelEventQueueType == SkQueueType::UNKNOWN) {
            SK_LOGE("%s node %zu failed to resolve previous KERNEL node and fallback queue type is UNKNOWN, nodeId=%lu",
                    to_string(nodeType), i, tasks[i]->GetNodeId());
            return false;
          }
          taskSyncInfos_[i].queueType = firstKernelEventQueueType;
          SK_LOGI("%s node %zu unable to resolve previous KERNEL node, nodeId=%lu, fallback = %s", to_string(nodeType),
                  i, tasks[i]->GetNodeId(), to_string(taskSyncInfos_[i].queueType));
        } else {
          kernelNodeCache[tasks[i]->GetNodeId()] = kernel;
          taskSyncInfos_[i].queueType =
              ToEventQueueType(ToQueueType(GetKernelInfos(kernel).kernelType), aicAvailable_, aivAvailable_);
        }
        resetCount++;
        break;
      }
      case SkNodeType::NODE_WAIT: {
        // WAIT node: inherit type from next KERNEL node
        auto *kernel =
            FindKernelNodeInDirection(tasks[i]->GetNextNodeId(), graph_, SearchDirection::NEXT, kernelNodeCache);
        if (kernel == nullptr) {
          if (firstKernelEventQueueType == SkQueueType::UNKNOWN) {
            SK_LOGE("%s node %zu failed to resolve next KERNEL node and fallback queue type is UNKNOWN, nodeId=%lu",
                    to_string(nodeType), i, tasks[i]->GetNodeId());
            return false;
          }
          taskSyncInfos_[i].queueType = firstKernelEventQueueType;
          SK_LOGI("%s node %zu unable to resolve next KERNEL node, nodeId=%lu, fallback = %s", to_string(nodeType), i,
                  tasks[i]->GetNodeId(), to_string(taskSyncInfos_[i].queueType));
        } else {
          kernelNodeCache[tasks[i]->GetNodeId()] = kernel;  // cache current WAIT node for future searches
          // Event nodes are executed on a single queue selected by nearest kernel type.
          SkQueueType nextKernelQueueType = ToQueueType(GetKernelInfos(kernel).kernelType);
          if (nextKernelQueueType != SkQueueType::AIC && nextKernelQueueType != SkQueueType::AIV) {
            taskSyncInfos_[i].queueType =
                ToEventQueueType(ResolveMixWaitEventQueueType(prevKernelTask, kernel), aicAvailable_, aivAvailable_);
          } else {
            taskSyncInfos_[i].queueType = ToEventQueueType(nextKernelQueueType, aicAvailable_, aivAvailable_);
          }
        }
        waitCount++;
        break;
      }
      default:
        SK_LOGE("unsupported node type for sync info initialization");
        return false;
    }
    if (nodeType != SkNodeType::NODE_NOTIFY && nodeType != SkNodeType::NODE_RESET && i < tasks.size() - 1) {
      // Initialize default cross-sync hints for AIC/AIV.
      switch (taskSyncInfos_[i].queueType) {
        case SkQueueType::AIC:
          taskSyncInfos_[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::CUB_TO_CUB;
          break;
        case SkQueueType::AIV:
          taskSyncInfos_[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIV)] = SyncDirection::VEC_TO_VEC;
          break;
        case SkQueueType::MIX_1_1:
        case SkQueueType::MIX_1_2:
          taskSyncInfos_[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::CUB_TO_CUB;
          taskSyncInfos_[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIV)] = SyncDirection::VEC_TO_VEC;
          break;
        default:
          SK_LOGE("unsupported kernel type : %s for inter-sync.", to_string(taskSyncInfos_[i].queueType));
          return false;
      }
    }
  }

  SK_LOGI("Initialized TaskSyncInfos for %zu tasks (%zu KERNEL, %zu NOTIFY, %zu WAIT, %zu RESET)", tasks.size(),
          kernelCount, notifyCount, waitCount, resetCount);
  return true;
}

// ========== Core sync insertion (aligned with Python insert_sync_event) ==========

void SkTaskBuilder::InsertSyncEvent(size_t preIdx, size_t currIdx) {
  SkQueueType preType = taskSyncInfos_[preIdx].queueType;
  SkQueueType currType = taskSyncInfos_[currIdx].queueType;
  // Generate sync direction between producer and consumer.
  SyncDirection dir = GenSyncDirection(preType, currType);

  SK_LOGI("  InsertSyncEvent: task[%zu](%s)(nodeId=%lu) -> task[%zu](%s)(nodeId=%lu), dir=%s", preIdx,
          to_string(preType), indexToNodeId_[preIdx], currIdx, to_string(currType), indexToNodeId_[currIdx],
          to_string(dir));

  // Sender side (preIdx): record send relation in the corresponding queue metadata.
  // CUB_TO_VEC: send SET in cub queue, receive WAIT in vec queue.
  // VEC_TO_CUB: send SET in vec queue, receive WAIT in cub queue.
  // MIX_TO_MIX: both directions are required.
  if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::MIX_TO_MIX) {
    taskSyncInfos_[preIdx].cubSendInfo[currIdx] = SyncDirection::CUB_TO_VEC;
  }
  if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::MIX_TO_MIX) {
    taskSyncInfos_[preIdx].vecSendInfo[currIdx] = SyncDirection::VEC_TO_CUB;
  }

  // Receiver side (currIdx): record recv relation in the corresponding queue metadata.
  if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::MIX_TO_MIX) {
    taskSyncInfos_[currIdx].vecRecvInfo[preIdx] = SyncDirection::CUB_TO_VEC;
  }
  if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::MIX_TO_MIX) {
    taskSyncInfos_[currIdx].cubRecvInfo[preIdx] = SyncDirection::VEC_TO_CUB;
  }
}

// ========== Print sync metadata (debug only) ==========

void SkTaskBuilder::PrintSyncInfo(const char *stage) const {
  SK_LOGI("%s", stage);
  SK_LOGI("[VEC LIST OP]:");
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];
    if (UsesAiv(info.queueType)) {
      std::string vecSendStr = "vecSend={";
      for (auto &kv : info.vecSendInfo) {
        vecSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      vecSendStr += "}, vecRecv={";
      for (auto &kv : info.vecRecvInfo) {
        vecSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      vecSendStr += "}";
      SK_LOGI("  task[%zu](%s): %s", i, to_string(info.queueType), vecSendStr.c_str());
    }
  }
  SK_LOGI("[CUB LIST OP]:");
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];
    if (UsesAic(info.queueType)) {
      std::string cubSendStr = "cubSend={";
      for (auto &kv : info.cubSendInfo) {
        cubSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      cubSendStr += "}, cubRecv={";
      for (auto &kv : info.cubRecvInfo) {
        cubSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      cubSendStr += "}";
      SK_LOGI("  task[%zu](%s): %s", i, to_string(info.queueType), cubSendStr.c_str());
    }
  }
}

// ========== Sync relation extraction ==========

bool SkTaskBuilder::PrecomputeSyncRelationsFromGraph(const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGI("Precomputing sync relations from graph, taskCount=%zu", tasks.size());

  // Initialize taskSyncInfos_.
  if (!InitTaskSyncInfos(tasks)) {
    SK_LOGE("PrecomputeSyncRelationsFromGraph failed: InitTaskSyncInfos failed");
    return false;
  }

  // 1. Extract intra-stream sync relations.
  SK_LOGI("[Sync by stream idx]");
  ExtractIntraStreamSync(tasks);
  // PrintSyncInfo("[After Sync by stream idx]");

  // 2. Extract inter-stream sync relations based on events.
  SK_LOGI("[Sync by event]");
  bool flag = ExtractInterStreamSync(tasks);
  // PrintSyncInfo("[After Sync by event]");
  return flag;
}

bool SkTaskBuilder::IsMixKernelTask(const SuperKernelBaseNode *task) const {
  if (task == nullptr) {
    return false;
  }
  if (task->GetNodeType() != SkNodeType::NODE_KERNEL) {
    return false;
  }
  return GetKernelInfos(task).needMixKernelSplit;
}

bool SkTaskBuilder::SplitTasksByMixGroups(const std::vector<SuperKernelBaseNode *> &tasks,
                                          std::vector<std::vector<SuperKernelBaseNode *>> &splitTasks,
                                          bool &hasMixKernel) const {
  splitTasks.clear();
  std::vector<SuperKernelBaseNode *> subTasks;
  bool inMixGroup = false;
  hasMixKernel = false;

  for (auto *task : tasks) {
    if (task == nullptr) {
      SK_LOGE("SplitTasksByMixGroups failed: null task found");
      return false;
    }
    const bool curIsMix = IsMixKernelTask(task);
    hasMixKernel = hasMixKernel || curIsMix;

    if (!subTasks.empty() && curIsMix != inMixGroup) {
      splitTasks.push_back(std::move(subTasks));
      subTasks.clear();
    }

    subTasks.push_back(task);
    inMixGroup = curIsMix;
  }

  if (!subTasks.empty()) {
    splitTasks.push_back(std::move(subTasks));
  }
  return true;
}

bool SkTaskBuilder::InitSyncInfoSnapshotForMixGroups(const std::vector<SuperKernelBaseNode *> &tasks,
                                                     std::vector<TaskSyncInfo> &taskSyncInfosOrigin) {
  // Initialize queue types from complete tasks. Sliced tasks may not resolve WAIT/NOTIFY queue type correctly.
  if (!InitTaskSyncInfos(tasks)) {
    SK_LOGE("InitSyncInfoSnapshotForMixGroups failed: InitTaskSyncInfos failed");
    return false;
  }
  taskSyncInfosOrigin = taskSyncInfos_;
  taskSyncInfos_.clear();
  if (taskSyncInfosOrigin.size() != tasks.size()) {
    SK_LOGE("InitSyncInfoSnapshotForMixGroups failed: init sync info size mismatch, origin=%zu, expected=%zu",
            taskSyncInfosOrigin.size(), tasks.size());
    return false;
  }
  return true;
}

bool SkTaskBuilder::RebaseTaskSyncInfo(TaskSyncInfo &syncInfo, size_t offset) const {
  auto rebaseSyncInfo = [](std::map<size_t, SyncDirection> &syncInfoMap, size_t offsetValue) -> bool {
    if (offsetValue == 0 || syncInfoMap.empty()) {
      return true;
    }
    std::map<size_t, SyncDirection> rebasedSyncInfo;
    for (const auto &syncPair : syncInfoMap) {
      if (syncPair.first > std::numeric_limits<size_t>::max() - offsetValue) {
        SK_LOGE("RebaseTaskSyncInfo failed: sync index overflow, index=%zu, offset=%zu", syncPair.first, offsetValue);
        return false;
      }
      rebasedSyncInfo[syncPair.first + offsetValue] = syncPair.second;
    }
    syncInfoMap.swap(rebasedSyncInfo);
    return true;
  };

  return rebaseSyncInfo(syncInfo.vecRecvInfo, offset) && rebaseSyncInfo(syncInfo.cubRecvInfo, offset) &&
         rebaseSyncInfo(syncInfo.vecSendInfo, offset) && rebaseSyncInfo(syncInfo.cubSendInfo, offset);
}

void SkTaskBuilder::AddBoundaryAllSync(const std::vector<SuperKernelBaseNode *> &curSplitTasks, size_t groupIndex,
                                       size_t groupOffset) {
  auto &boundarySyncInfo = taskSyncInfos_.back();
  SK_LOGI(
      "AddBoundaryAllSync: add boundary all-sync at group=%zu, localTask=%zu, "
      "globalTask=%zu, nodeId=%lu",
      groupIndex, curSplitTasks.size() - 1, groupOffset + curSplitTasks.size() - 1, curSplitTasks.back()->GetNodeId());
  boundarySyncInfo.vecSendInfo.clear();
  boundarySyncInfo.cubSendInfo.clear();
  boundarySyncInfo.crossSyncInfo.clear();
  boundarySyncInfo.crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::ALL_SYNC;
}

bool SkTaskBuilder::ProcessSyncRelationSplitGroup(const std::vector<SuperKernelBaseNode *> &curSplitTasks,
                                                  size_t groupIndex, size_t groupOffset, bool hasNextGroup,
                                                  const std::vector<TaskSyncInfo> &taskSyncInfosOrigin,
                                                  std::vector<TaskSyncInfo> &mergedTaskSyncInfos) {
  if (curSplitTasks.empty()) {
    SK_LOGE("ProcessSyncRelationSplitGroup failed: empty split group, index=%zu", groupIndex);
    return false;
  }
  if (groupOffset > taskSyncInfosOrigin.size() || curSplitTasks.size() > taskSyncInfosOrigin.size() - groupOffset) {
    SK_LOGE(
        "ProcessSyncRelationSplitGroup failed: split range out of bounds, index=%zu, offset=%zu, "
        "groupSize=%zu, originSize=%zu",
        groupIndex, groupOffset, curSplitTasks.size(), taskSyncInfosOrigin.size());
    return false;
  }

  const bool groupIsMix = IsMixKernelTask(curSplitTasks.front());
  SK_LOGI(
      "ProcessSyncRelationSplitGroup: process group index=%zu, offset=%zu, size=%zu, isMixGroup=%d, "
      "firstNodeId=%lu, lastNodeId=%lu",
      groupIndex, groupOffset, curSplitTasks.size(), static_cast<int>(groupIsMix), curSplitTasks.front()->GetNodeId(),
      curSplitTasks.back()->GetNodeId());

  taskSyncInfos_.clear();
  nodeIdToIndex_.clear();
  indexToNodeId_.clear();
  taskSyncInfos_.resize(curSplitTasks.size());
  for (size_t j = 0; j < curSplitTasks.size(); j++) {
    taskSyncInfos_[j] = taskSyncInfosOrigin[groupOffset + j];
  }

  SK_LOGI("Build sub-phase-1 begin: precompute sync relations, taskCount=%zu, index=%zu", curSplitTasks.size(),
          groupIndex);
  SK_LOGI("[sub sync by stream idx]");
  ExtractIntraStreamSync(curSplitTasks);

  SK_LOGI("[sub sync by event]");
  if (!ExtractInterStreamSync(curSplitTasks)) {
    SK_LOGE("ProcessSyncRelationSplitGroup failed: ExtractInterStreamSync failed");
    return false;
  }

  SK_LOGI("Build sub-phase-2 begin: optimize sync relations, index=%zu", groupIndex);
  OptimizeSyncRelations(curSplitTasks);

  if (hasNextGroup) {
    AddBoundaryAllSync(curSplitTasks, groupIndex, groupOffset);
  }
  for (auto &syncInfo : taskSyncInfos_) {
    if (!RebaseTaskSyncInfo(syncInfo, groupOffset)) {
      return false;
    }
    mergedTaskSyncInfos.push_back(std::move(syncInfo));
  }

  SK_LOGI("ProcessSyncRelationSplitGroup: finish group index=%zu, mergedTaskCount=%zu", groupIndex,
          mergedTaskSyncInfos.size());
  return true;
}

bool SkTaskBuilder::PrecomputeSyncRelationsByMixGroups(const std::vector<SuperKernelBaseNode *> &tasks) {
  if (tasks.empty()) {
    SK_LOGE("PrecomputeSyncRelationsByMixGroups failed: empty tasks");
    return false;
  }

  std::vector<std::vector<SuperKernelBaseNode *>> splitTasks;
  bool hasMixKernel = false;
  if (!SplitTasksByMixGroups(tasks, splitTasks, hasMixKernel)) {
    return false;
  }
  SK_LOGI("PrecomputeSyncRelationsByMixGroups: taskCount=%zu, splitGroupCount=%zu, hasMixKernel=%d", tasks.size(),
          splitTasks.size(), static_cast<int>(hasMixKernel));

  std::vector<TaskSyncInfo> taskSyncInfosOrigin;
  if (!InitSyncInfoSnapshotForMixGroups(tasks, taskSyncInfosOrigin)) {
    return false;
  }
  std::vector<TaskSyncInfo> mergedTaskSyncInfos;
  mergedTaskSyncInfos.reserve(tasks.size());
  size_t curTaskCount = 0;
  for (size_t i = 0; i < splitTasks.size(); i++) {
    if (!ProcessSyncRelationSplitGroup(splitTasks[i], i, curTaskCount, i + 1 < splitTasks.size(), taskSyncInfosOrigin,
                                       mergedTaskSyncInfos)) {
      return false;
    }
    curTaskCount += splitTasks[i].size();
  }

  if (curTaskCount != tasks.size() || mergedTaskSyncInfos.size() != tasks.size()) {
    SK_LOGE(
        "PrecomputeSyncRelationsByMixGroups failed: merged task sync info size mismatch, consumed=%zu, "
        "merged=%zu, expected=%zu",
        curTaskCount, mergedTaskSyncInfos.size(), tasks.size());
    return false;
  }
  taskSyncInfos_.swap(mergedTaskSyncInfos);
  nodeIdToIndex_.clear();
  indexToNodeId_.clear();
  SK_LOGI("PrecomputeSyncRelationsByMixGroups complete: taskCount=%zu, splitGroupCount=%zu", tasks.size(),
          splitTasks.size());
  return true;
}

// label : success
void SkTaskBuilder::ExtractIntraStreamSync(const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGI("ExtractIntraStreamSync: processing %zu tasks", tasks.size());

  // Group tasks by stream (aligned with Python insert_sync_by_stream_idx).
  std::map<uint32_t, std::vector<size_t>> streamOps;
  for (size_t i = 0; i < tasks.size(); i++) {
    uint32_t streamIdx = tasks[i]->GetStreamIdxInGraph();
    streamOps[streamIdx].push_back(i);
    nodeIdToIndex_[tasks[i]->GetNodeId()] = i;
    indexToNodeId_[i] = tasks[i]->GetNodeId();
  }

  auto streamfusionOption = opts.GetOption(aclskOptionType::STREAM_FUSION);
  uint32_t streamFusionValue = 1;
  if (streamfusionOption != nullptr) {
    streamFusionValue = streamfusionOption->GetIntValue();
  }
  if (streamFusionValue == 0 && streamOps.size() > 1) {
    SK_LOGW(
        "Multi stream fusion is triggered with %zu streams detected, "
        "but aclskStreamFusionOption is off (value=%u). "
        "To explicitly enable multi stream fusion, set aclskStreamFusionOption to 1. "
        "Please confirm whether this fusion behavior meets your expectations.",
        streamOps.size(), streamFusionValue);
  }

  // Insert sync for consecutive tasks in each stream.
  size_t totalInsertedSyncEdges = 0;
  for (auto &streamPair : streamOps) {
    SK_LOGI("Process stream sync: streamIdx=%u, opCount=%zu", streamPair.first, streamPair.second.size());
    auto &opList = streamPair.second;
    for (size_t j = 1; j < opList.size(); j++) {
      size_t preIdx = opList[j - 1];
      size_t currIdx = opList[j];
      InsertSyncEvent(preIdx, currIdx);
      ++totalInsertedSyncEdges;
    }
  }
  SK_LOGI("ExtractIntraStreamSync complete: streamCount=%zu, insertedSyncEdges=%zu", streamOps.size(),
          totalInsertedSyncEdges);
}

/**
 * @brief Extract inter-stream synchronization relationships based on task dependencies
 *
 * This function identifies synchronization requirements between different execution streams
 * by analyzing task dependencies. It inserts sync events (NOTIFY/WAIT pairs) to establish
 * proper synchronization boundaries when tasks from different streams need to coordinate.
 *
 * @param tasks Vector of super kernel base nodes to analyze
 *
 * @note This function works in two phases:
 *       1. Build a task ID set for quick lookup
 *       2. For each KERNEL node, check its successors and insert sync events
 *          when the successor is also in the current task set
 */
bool SkTaskBuilder::ExtractInterStreamSync(const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGI("ExtractInterStreamSync: starting to extract inter-stream synchronization relationships");
  SK_LOGI("ExtractInterStreamSync: total number of tasks to process = %zu", tasks.size());

  // Aligned with Python insert_sync_by_event behavior.
  // In C++, NOTIFY/WAIT are standalone nodes mapped to NotifyFunc/WaitFunc.
  // Sync relation is modeled as: NOTIFY node -> WAIT node.

  // Phase 1: Build task ID set for O(1) lookup
  std::unordered_set<uint64_t> taskIds;
  for (size_t i = 0; i < tasks.size(); i++) {
    SuperKernelBaseNode *node = tasks[i];
    taskIds.insert(node->GetNodeId());
  }
  SK_LOGI("ExtractInterStreamSync: built task ID set with %zu unique IDs", taskIds.size());

  // Phase 2: Traverse tasks and insert sync events for KERNEL nodes
  uint32_t syncEventCount = 0;
  for (size_t i = 0; i < tasks.size(); i++) {
    SuperKernelBaseNode *node = tasks[i];
    auto preNodeId = node->GetNodeId();
    SkNodeType nodeType = node->GetNodeType();
    // Only process KERNEL nodes for synchronization
    if (nodeType == SkNodeType::NODE_KERNEL) {
      SK_LOGI("ExtractInterStreamSync: processing KERNEL node %lu with %zu successors", preNodeId,
              node->sendToNodeId.size());
      // Check each successor to determine if sync event is needed
      for (auto nextId : node->sendToNodeId) {
        // If successor is in the current task set, insert sync event
        if (taskIds.find(nextId) != taskIds.end()) {
          SK_LOGI("ExtractInterStreamSync: inserting sync event between %lu -> %lu", preNodeId, nextId);
          if (nodeIdToIndex_.find(preNodeId) == nodeIdToIndex_.end() ||
              nodeIdToIndex_.find(nextId) == nodeIdToIndex_.end()) {
            SK_LOGE("NodeId not exists in task nodes, node id is %lu, %lu", preNodeId, nextId);
            return false;
          }
          InsertSyncEvent(nodeIdToIndex_[preNodeId], nodeIdToIndex_[nextId]);
          syncEventCount++;
        } else {
          SK_LOGI("ExtractInterStreamSync: skipping external successor %lu (not in task set)", nextId);
        }
      }
    }
  }

  SK_LOGI("ExtractInterStreamSync: completed, inserted %u sync events", syncEventCount);
  return true;
}

// ========== Sync optimization (aligned with Python behavior) ==========

void SkTaskBuilder::OptimizeSyncRelations(const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGI("Optimizing Sync Relations");
  // PrintSyncInfo("[INIT STATE]");

  RemoveCrossedLineSync();
  // PrintSyncInfo("[AFTER REMOVE CROSSED LINE SYNC]");

  RemoveMultiSendSync();
  RemoveMultiRecvSync();
  // PrintSyncInfo("[AFTER REMOVE MULTI EVENT SYNC]");

  RemoveRedundantCrossSync(tasks);
}

bool SkTaskBuilder::JudgeRemoveCrossSync(size_t sendIdx, size_t recvIdx, bool isCubToVec) {
  // Equivalent to Python judge_remove.
  // Check whether there is a crossed sync edge.
  // Crossing condition:
  // another edge (send_idx1 -> recv_idx1) exists where recv_idx1 < recv_idx and send_idx1 > send_idx.

  if (isCubToVec) {
    // cub:vec direction
    for (size_t otherRecvIdx = 0; otherRecvIdx < recvIdx; otherRecvIdx++) {
      // Inspect receiver metadata.
      for (auto &kv : taskSyncInfos_[otherRecvIdx].vecRecvInfo) {
        size_t otherSendIdx = kv.first;
        SyncDirection dir = kv.second;
        if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::MIX_TO_MIX) {
          // otherRecvIdx < recvIdx is guaranteed by loop condition.
          // Only verify otherSendIdx > sendIdx.
          if (otherSendIdx > sendIdx) {
            SK_LOGI("  Found crossed sync: task[%zu]->task[%zu] crosses with task[%zu]->task[%zu]", otherSendIdx,
                    otherRecvIdx, sendIdx, recvIdx);
            return true;
          }
        }
      }
    }
  } else {
    // vec:cub direction
    for (size_t otherRecvIdx = 0; otherRecvIdx < recvIdx; otherRecvIdx++) {
      // Inspect receiver metadata.
      for (auto &kv : taskSyncInfos_[otherRecvIdx].cubRecvInfo) {
        size_t otherSendIdx = kv.first;
        SyncDirection dir = kv.second;
        if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::MIX_TO_MIX) {
          // otherRecvIdx < recvIdx is guaranteed by loop condition.
          // Only verify otherSendIdx > sendIdx.
          if (otherSendIdx > sendIdx) {
            SK_LOGI("  Found crossed sync: task[%zu]->task[%zu] crosses with task[%zu]->task[%zu]", otherSendIdx,
                    otherRecvIdx, sendIdx, recvIdx);
            return true;
          }
        }
      }
    }
  }
  return false;
}

void SkTaskBuilder::RemoveSyncInfo(size_t sendIdx, size_t recvIdx, bool isRemoveRecv, SyncDirection dirToRemove) {
  // Equivalent to Python remove_info_by_name.
  auto removeDirection = [](std::map<size_t, SyncDirection> &info, size_t key, SyncDirection toRemove) {
    auto it = info.find(key);
    if (it != info.end() && it->second == toRemove) {
      info.erase(it);
    }
  };

  if (isRemoveRecv) {
    removeDirection(taskSyncInfos_[sendIdx].vecRecvInfo, recvIdx, dirToRemove);
    removeDirection(taskSyncInfos_[sendIdx].cubRecvInfo, recvIdx, dirToRemove);
  } else {
    removeDirection(taskSyncInfos_[sendIdx].vecSendInfo, recvIdx, dirToRemove);
    removeDirection(taskSyncInfos_[sendIdx].cubSendInfo, recvIdx, dirToRemove);
  }
}

void SkTaskBuilder::RemoveCrossedLineSync() {
  SK_LOGI("RemoveCrossedLineSync: checking crossed sync relations");

  // Check crossed sync for cub queue (CUB_TO_VEC direction).
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];

    std::vector<size_t> toRemove;
    for (auto &kv : info.cubSendInfo) {
      if (kv.second == SyncDirection::CUB_TO_VEC && JudgeRemoveCrossSync(i, kv.first, true)) {
        toRemove.push_back(kv.first);
      }
    }
    for (size_t recvIdx : toRemove) {
      SK_LOGI("  Remove crossed cub:vec sync: task[%zu] -> task[%zu]", i, recvIdx);
      RemoveSyncInfo(i, recvIdx, false, SyncDirection::CUB_TO_VEC);
      RemoveSyncInfo(recvIdx, i, true, SyncDirection::CUB_TO_VEC);
    }
  }

  // Check crossed sync for vec queue (VEC_TO_CUB direction).
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];

    std::vector<size_t> toRemove;
    for (auto &kv : info.vecSendInfo) {
      if (kv.second == SyncDirection::VEC_TO_CUB && JudgeRemoveCrossSync(i, kv.first, false)) {
        toRemove.push_back(kv.first);
      }
    }
    for (size_t recvIdx : toRemove) {
      SK_LOGI("  Remove crossed vec:cub sync: task[%zu] -> task[%zu]", i, recvIdx);
      RemoveSyncInfo(i, recvIdx, false, SyncDirection::VEC_TO_CUB);
      RemoveSyncInfo(recvIdx, i, true, SyncDirection::VEC_TO_CUB);
    }
  }
}

void SkTaskBuilder::RemoveMultiSendSync() {
  SK_LOGI("RemoveMultiSendSync: removing redundant send sync");

  // Check vec queue (VEC_TO_CUB direction).
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];
    if (info.vecSendInfo.size() > 1) {
      std::vector<size_t> vecSendList;
      for (auto &kv : info.vecSendInfo) {
        if (kv.second == SyncDirection::VEC_TO_CUB) {
          vecSendList.push_back(kv.first);
        }
      }
      if (vecSendList.size() > 1) {
        std::sort(vecSendList.begin(), vecSendList.end());
        for (size_t j = 1; j < vecSendList.size(); j++) {
          size_t recvIdx = vecSendList[j];
          SK_LOGI("  Remove multi vec:cub send sync: task[%zu] -> task[%zu]", i, recvIdx);
          RemoveSyncInfo(i, recvIdx, false, SyncDirection::VEC_TO_CUB);
          RemoveSyncInfo(recvIdx, i, true, SyncDirection::VEC_TO_CUB);
        }
      }
    }
  }

  // Check cub queue (CUB_TO_VEC direction).
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];
    if (info.cubSendInfo.size() > 1) {
      std::vector<size_t> cubSendList;
      for (auto &kv : info.cubSendInfo) {
        if (kv.second == SyncDirection::CUB_TO_VEC) {
          cubSendList.push_back(kv.first);
        }
      }
      if (cubSendList.size() > 1) {
        std::sort(cubSendList.begin(), cubSendList.end());
        for (size_t j = 1; j < cubSendList.size(); j++) {
          size_t recvIdx = cubSendList[j];
          SK_LOGI("  Remove multi cub:vec send sync: task[%zu] -> task[%zu]", i, recvIdx);
          RemoveSyncInfo(i, recvIdx, false, SyncDirection::CUB_TO_VEC);
          RemoveSyncInfo(recvIdx, i, true, SyncDirection::CUB_TO_VEC);
        }
      }
    }
  }
}

void SkTaskBuilder::RemoveMultiRecvSync() {
  SK_LOGI("RemoveMultiRecvSync: removing redundant recv sync");

  // Check vec queue (CUB_TO_VEC direction).
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];
    if (info.vecRecvInfo.size() > 1) {
      std::vector<size_t> vecRecvList;
      for (auto &kv : info.vecRecvInfo) {
        if (kv.second == SyncDirection::CUB_TO_VEC) {
          vecRecvList.push_back(kv.first);
        }
      }
      if (vecRecvList.size() > 1) {
        // Sort in descending index order and keep the latest receiver.
        std::sort(vecRecvList.begin(), vecRecvList.end(), std::greater<>());
        for (size_t j = 1; j < vecRecvList.size(); j++) {
          size_t sendIdx = vecRecvList[j];
          SK_LOGI("  Remove multi cub:vec recv sync: task[%zu] <- task[%zu]", i, sendIdx);
          RemoveSyncInfo(i, sendIdx, true, SyncDirection::CUB_TO_VEC);
          RemoveSyncInfo(sendIdx, i, false, SyncDirection::CUB_TO_VEC);
        }
      }
    }
  }

  // Check cub queue (VEC_TO_CUB direction).
  for (size_t i = 0; i < taskSyncInfos_.size(); i++) {
    auto &info = taskSyncInfos_[i];
    if (info.cubRecvInfo.size() > 1) {
      std::vector<size_t> cubRecvList;
      for (auto &kv : info.cubRecvInfo) {
        if (kv.second == SyncDirection::VEC_TO_CUB) {
          cubRecvList.push_back(kv.first);
        }
      }
      if (cubRecvList.size() > 1) {
        std::sort(cubRecvList.begin(), cubRecvList.end(), std::greater<>());
        for (size_t j = 1; j < cubRecvList.size(); j++) {
          size_t sendIdx = cubRecvList[j];
          SK_LOGI("  Remove multi vec:cub recv sync: task[%zu] <- task[%zu]", i, sendIdx);
          RemoveSyncInfo(i, sendIdx, true, SyncDirection::VEC_TO_CUB);
          RemoveSyncInfo(sendIdx, i, false, SyncDirection::VEC_TO_CUB);
        }
      }
    }
  }
}

namespace {
size_t PickCandidateKernelIdx(size_t lastAicKernelIdx, size_t lastAivKernelIdx, SkQueueType queueType,
                              size_t invalidIdx) {
  size_t candidateIdx = invalidIdx;
  if (queueType == SkQueueType::AIC) {
    candidateIdx = lastAicKernelIdx;
  } else if (queueType == SkQueueType::AIV) {
    candidateIdx = lastAivKernelIdx;
  } else {
    SK_LOGE("Unexpected queue type %s for WAIT node, cannot determine candidate kernel for redundant sync removal.",
            to_string(queueType));
    return invalidIdx;
  }
  return candidateIdx;
}
}  // namespace

void SkTaskBuilder::RemoveRedundantCrossSync(const std::vector<SuperKernelBaseNode *> &tasks) {
  const size_t invalidIdx = static_cast<size_t>(INVALID_TASK_ID);
  size_t lastAicKernelIdx = invalidIdx;
  size_t lastAivKernelIdx = invalidIdx;

  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto nodeType = tasks[i]->GetNodeType();
    auto &info = taskSyncInfos_[i];
    if (nodeType == SkNodeType::NODE_KERNEL) {
      if (UsesAic(info.queueType)) {
        lastAicKernelIdx = i;
      }
      if (UsesAiv(info.queueType)) {
        lastAivKernelIdx = i;
      }
      continue;
    }
    if (nodeType != SkNodeType::NODE_WAIT) {
      continue;
    }
    const size_t candidateKernelIdx =
        PickCandidateKernelIdx(lastAicKernelIdx, lastAivKernelIdx, info.queueType, invalidIdx);
    if (candidateKernelIdx == invalidIdx) {
      continue;
    }

    auto &candidateInfo = taskSyncInfos_[candidateKernelIdx];
    if (info.queueType == SkQueueType::AIC && candidateInfo.cubSendInfo.empty()) {
      candidateInfo.crossSyncInfo.erase(static_cast<size_t>(SkQueueType::AIC));
      SK_LOGI("Remove redundant cross sync: task[%zu] -> task[%zu], remove CUB_TO_CUB hint", candidateKernelIdx, i);
    } else if (info.queueType == SkQueueType::AIV && candidateInfo.vecSendInfo.empty()) {
      candidateInfo.crossSyncInfo.erase(static_cast<size_t>(SkQueueType::AIV));
      SK_LOGI("Remove redundant cross sync: task[%zu] -> task[%zu], remove VEC_TO_VEC hint", candidateKernelIdx, i);
    } else {
      SK_LOGI("No redundant cross sync to remove for task[%zu] -> task[%zu]", candidateKernelIdx, i);
    }
  }
}

namespace {
void ApplySplitCoreControlMask(SuperKernelBaseNode *task, TaskSyncInfo &taskSyncInfo) {
  if (GetSkRuntimeConfig().kernelArch != SkKernelArch::DAV_3510) {
    return;
  }
  if (task->GetNodeType() != SkNodeType::NODE_KERNEL) {
    return;
  }
  taskSyncInfo.earlyStartInfo.ApplyFuncMask(SkEarlyStartMask::SPLIT_CORE_CTRL);
  SK_LOGI("early-start control: nodeId=%lu, set split-core ctrl mask", task->GetNodeId());
}

bool ApplyEarlyStartSyncForQueue(const std::vector<SuperKernelBaseNode *> &tasks,
                                 std::vector<TaskSyncInfo> &taskSyncInfos, size_t curIdx, size_t &lastIdx,
                                 SkQueueType targetQueue) {
  if (tasks[curIdx]->GetNodeType() != SkNodeType::NODE_KERNEL) {
    lastIdx = curIdx;
    return true;
  }

  const bool isAic = targetQueue == SkQueueType::AIC;
  const char *sendInfoName = isAic ? "cubSend" : "vecSend";
  const char *recvInfoName = isAic ? "cubRecv" : "vecRecv";
  const auto queueTypeId = static_cast<size_t>(targetQueue);
  const auto selfSyncDir = isAic ? SyncDirection::CUB_TO_CUB : SyncDirection::VEC_TO_VEC;
  const auto selfSetMask = isAic ? SkEarlyStartMask::AIC_TO_AIC_SET : SkEarlyStartMask::AIV_TO_AIV_SET;
  const auto selfWaitMask = isAic ? SkEarlyStartMask::AIC_TO_AIC_WAIT : SkEarlyStartMask::AIV_TO_AIV_WAIT;
  const auto crossSendMask = isAic ? SkEarlyStartMask::AIC_TO_AIV_SET : SkEarlyStartMask::AIV_TO_AIC_SET;
  const auto crossRecvMask = isAic ? SkEarlyStartMask::AIV_TO_AIC_WAIT : SkEarlyStartMask::AIC_TO_AIV_WAIT;

  const KernelInfos &kernelInfo = tasks[curIdx]->GetNodeInfos().kernelInfos;
  auto &curSyncInfo = taskSyncInfos[curIdx];
  if (kernelInfo.capBits.earlyStartSetFlag) {
    auto curCrossIt = curSyncInfo.crossSyncInfo.find(queueTypeId);
    if (curCrossIt != curSyncInfo.crossSyncInfo.end() && curCrossIt->second == selfSyncDir) {
      curSyncInfo.earlyStartInfo.relatedSetNode = tasks[curIdx];
      curSyncInfo.earlyStartInfo.ApplyFuncMask(selfSetMask);
      curSyncInfo.earlyStartInfo.ApplySyncMask(selfSetMask);
      curSyncInfo.earlyStartInfo.ApplySyncMask(selfWaitMask);
      curSyncInfo.crossSyncInfo.erase(curCrossIt);
      SK_LOGI("early-start rewrite: slot=cur.crossSyncInfo[%s], task[%zu](nodeId=%lu), self sync",
              to_string(targetQueue), curIdx, tasks[curIdx]->GetNodeId());
    }
  }
  if (kernelInfo.capBits.earlyStartWaitFlag && lastIdx != static_cast<size_t>(INVALID_TASK_ID)) {
    auto &prevSyncInfo = taskSyncInfos[lastIdx];
    auto &prevSendInfo = isAic ? prevSyncInfo.cubSendInfo : prevSyncInfo.vecSendInfo;
    auto &curRecvInfo = isAic ? curSyncInfo.cubRecvInfo : curSyncInfo.vecRecvInfo;
    if (prevSendInfo.size() > 1) {
      SK_LOGE(
          "ApplyEarlyStartSyncPass failed: slot=prev.%s has multiple syncs, "
          "prevTask[%zu](nodeId=%lu), curTask[%zu](nodeId=%lu), %sSize=%zu",
          sendInfoName, lastIdx, tasks[lastIdx]->GetNodeId(), curIdx, tasks[curIdx]->GetNodeId(), sendInfoName,
          prevSendInfo.size());
      return false;
    }
    if (curRecvInfo.size() > 1) {
      SK_LOGE(
          "ApplyEarlyStartSyncPass failed: slot=cur.%s has multiple syncs, "
          "prevTask[%zu](nodeId=%lu), curTask[%zu](nodeId=%lu), %sSize=%zu",
          recvInfoName, lastIdx, tasks[lastIdx]->GetNodeId(), curIdx, tasks[curIdx]->GetNodeId(), recvInfoName,
          curRecvInfo.size());
      return false;
    }
    if (isAic) {
      prevSyncInfo.earlyStartInfo.nextAicRelatedNode = tasks[curIdx];
    } else {
      prevSyncInfo.earlyStartInfo.nextAivRelatedNode = tasks[curIdx];
    }
    curSyncInfo.earlyStartInfo.relatedWaitNode = tasks[curIdx];
    auto prevCrossIt = prevSyncInfo.crossSyncInfo.find(queueTypeId);
    if (prevCrossIt != prevSyncInfo.crossSyncInfo.end() && prevCrossIt->second == selfSyncDir) {
      // case prev task not enable early-start
      prevSyncInfo.earlyStartInfo.ApplySyncMask(selfSetMask);
      prevSyncInfo.earlyStartInfo.ApplySyncMask(selfWaitMask);
      curSyncInfo.earlyStartInfo.ApplyFuncMask(selfWaitMask);
      prevSyncInfo.crossSyncInfo.erase(prevCrossIt);
      SK_LOGI(
          "early-start rewrite: slot=prev.crossSyncInfo[%s], prevTask[%zu](nodeId=%lu) -> "
          "curTask[%zu](nodeId=%lu), self sync",
          to_string(targetQueue), lastIdx, tasks[lastIdx]->GetNodeId(), curIdx, tasks[curIdx]->GetNodeId());
    } else if (prevSyncInfo.earlyStartInfo.CheckSyncMask(selfSetMask)) {
      // case prev task already enable early-start, but not set wait config
      curSyncInfo.earlyStartInfo.ApplyFuncMask(selfWaitMask);
      SK_LOGI("early-start attach: slot=prev.earlyStartInfo.syncConfig, curTask[%zu](nodeId=%lu), %s wait", curIdx,
              tasks[curIdx]->GetNodeId(), to_string(targetQueue));
    }
    if (!prevSendInfo.empty()) {
      SK_LOGI("early-start absorb: slot=prev.%s, prevTask[%zu](nodeId=%lu), curTask[%zu](nodeId=%lu)", sendInfoName,
              lastIdx, tasks[lastIdx]->GetNodeId(), curIdx, tasks[curIdx]->GetNodeId());
      prevSyncInfo.earlyStartInfo.ApplySyncMask(crossSendMask);
      curSyncInfo.earlyStartInfo.ApplyFuncMask(crossSendMask);
      prevSendInfo.clear();
    }
    if (!curRecvInfo.empty()) {
      SK_LOGI("early-start absorb: slot=cur.%s, curTask[%zu](nodeId=%lu)", recvInfoName, curIdx,
              tasks[curIdx]->GetNodeId());
      curSyncInfo.earlyStartInfo.ApplySyncMask(crossRecvMask);
      curSyncInfo.earlyStartInfo.ApplyFuncMask(crossRecvMask);
      curRecvInfo.clear();
    }
  }
  lastIdx = curIdx;
  return true;
}

void LogEarlyStartFinalState(const std::vector<SuperKernelBaseNode *> &tasks,
                             const std::vector<TaskSyncInfo> &taskSyncInfos) {
  const auto getNodeId = [](const SuperKernelBaseNode *node) -> uint64_t {
    return node == nullptr ? INVALID_TASK_ID : node->GetNodeId();
  };
  for (size_t i = 0; i < taskSyncInfos.size(); ++i) {
    const auto &earlyStartInfo = taskSyncInfos[i].earlyStartInfo;
    const uint64_t relatedSetNodeId = getNodeId(earlyStartInfo.relatedSetNode);
    const uint64_t relatedWaitNodeId = getNodeId(earlyStartInfo.relatedWaitNode);
    const uint64_t nextAicNodeId = getNodeId(earlyStartInfo.nextAicRelatedNode);
    const uint64_t nextAivNodeId = getNodeId(earlyStartInfo.nextAivRelatedNode);
    SK_LOGI(
        "early-start final: task[%zu](nodeId=%lu), funcConfig=0x%x, syncConfig=0x%x, "
        "relatedSetNodeId=%lu, relatedWaitNodeId=%lu, nextAicNodeId=%lu, nextAivNodeId=%lu",
        i, tasks[i]->GetNodeId(), earlyStartInfo.funcEarlyStartConfig, earlyStartInfo.syncEarlyStartConfig,
        relatedSetNodeId, relatedWaitNodeId, nextAicNodeId, nextAivNodeId);
  }
}

bool HasAllSyncBarrier(const TaskSyncInfo &taskSyncInfo) {
  for (const auto &syncInfo : taskSyncInfo.crossSyncInfo) {
    if (syncInfo.second == SyncDirection::ALL_SYNC) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool SkTaskBuilder::ApplyEarlyStartSyncPass(const std::vector<SuperKernelBaseNode *> &tasks) {
  SK_LOGI("ApplyEarlyStartSyncPass begin: taskCount=%zu", tasks.size());
  size_t lastAicTaskIdx = static_cast<size_t>(INVALID_TASK_ID);
  size_t lastAivTaskIdx = static_cast<size_t>(INVALID_TASK_ID);
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (UsesAic(taskSyncInfos_[i].queueType) &&
        !ApplyEarlyStartSyncForQueue(tasks, taskSyncInfos_, i, lastAicTaskIdx, SkQueueType::AIC)) {
      return false;
    }
    if (UsesAiv(taskSyncInfos_[i].queueType) &&
        !ApplyEarlyStartSyncForQueue(tasks, taskSyncInfos_, i, lastAivTaskIdx, SkQueueType::AIV)) {
      return false;
    }
    // DAV_3510 early-start uses split-core control for all kernel types.
    ApplySplitCoreControlMask(tasks[i], taskSyncInfos_[i]);
    if (HasAllSyncBarrier(taskSyncInfos_[i])) {
      lastAicTaskIdx = i;
      lastAivTaskIdx = i;
      SK_LOGI("early-start barrier: task[%zu](nodeId=%lu) has ALL_SYNC barrier.", i, tasks[i]->GetNodeId());
    }
  }
  LogEarlyStartFinalState(tasks, taskSyncInfos_);
  return true;
}

namespace {
SkCoreSyncType GetSyncTypesForDirection(SyncDirection dir, bool isSend) {
  SkCoreSyncType syncType = SkCoreSyncType::SYNC_NONE;

  // Note: this path normally receives only CUB_TO_VEC or VEC_TO_CUB.
  switch (dir) {
    case SyncDirection::CUB_TO_VEC:
      // AIC -> AIV
      syncType = isSend ? SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV : SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV;
      break;
    case SyncDirection::VEC_TO_CUB:
      // AIV -> AIC
      syncType = isSend ? SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC : SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC;
      break;
    case SyncDirection::CUB_TO_CUB:
      // AIC -> AIC
      syncType = SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC;
      break;
    case SyncDirection::VEC_TO_VEC:
      // AIV -> AIV
      syncType = SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV;
      break;
    case SyncDirection::MIX_TO_MIX:
      // Should not appear because InsertSyncEvent has split this case.
      syncType = SkCoreSyncType::SYNC_NONE;
      SK_LOGI("GetSyncTypesForDirection: unexpected MIX_TO_MIX direction, add syncType %s.", to_string(syncType));
      break;
    case SyncDirection::NONE:
      syncType = SkCoreSyncType::SYNC_NONE;
      break;
    case SyncDirection::ALL_SYNC:
      syncType = SkCoreSyncType::ALL_SYNC;
      break;
    default:
      syncType = SkCoreSyncType::SYNC_NONE;
      SK_LOGI("GetSyncTypesForDirection: unknown direction %s, add syncType %s.", to_string(dir), to_string(syncType));
      break;
  }
  return syncType;
}

}  // namespace

DeviceArgsPtr SkTaskBuilder::GenEntryArgs(const SkTask &skTaskCube, const SkTask &skTaskVec, const SkDfxInfo *dfxInfos,
                                          uint32_t dfxCount, const SkEventConfig *eventConfig) {
  size_t header_size = sizeof(SkHeaderInfo);
  size_t aic_que_size = skTaskCube.GetTaskQueSize();
  size_t aiv_que_size = skTaskVec.GetTaskQueSize();
  size_t counter_size = GetSkRuntimeConfig().eventCoreNum * sizeof(SkCounterInfo);
  size_t dfx_size = dfxCount * sizeof(SkDfxInfo);
  size_t event_config_size = sizeof(SkEventConfig);

  size_t aic_que_offset = header_size;
  size_t aiv_que_offset = aic_que_offset + aic_que_size * 4;
  size_t counter_offset = AlignUp(aiv_que_offset + aiv_que_size * 4, kCounterAlignBytes);
  size_t dfx_offset = counter_offset + counter_size;
  size_t event_config_offset = dfx_offset + dfx_size;
  uint64_t total_size = event_config_offset + event_config_size;

  SK_LOGI(
      "sk total args size: %lu, header_size: %zu, aic_que_size: %zu, aiv_que_size: %zu, "
      "counter_size: %zu, dfx_size: %zu, counter_offset: %zu",
      total_size, header_size, aic_que_size, aiv_que_size, counter_size, dfx_size, counter_offset);
  DeviceArgsPtr args;
  if (!args.Init(total_size)) {
    SK_LOGE("GenEntryArgs init device args failed, total_size=%lu", total_size);
    return {};
  }
  args.Get()->skHeader.aicQueSize = aic_que_size;
  args.Get()->skHeader.aivQueSize = aiv_que_size;
  args.Get()->skHeader.aicQueOffset = aic_que_offset;
  args.Get()->skHeader.aivQueOffset = aiv_que_offset;
  args.Get()->skHeader.counterOffset = counter_offset;
  args.Get()->skHeader.dfxOffset = dfx_offset;
  args.Get()->skHeader.eventConfigOffset = event_config_offset;
  args.Get()->skHeader.totalSize = total_size;

  uint8_t *base = (uint8_t *)args.Get();

  errno_t err = 0;
  // Copy AIC queues
  for (size_t i = 0; i < 4; i++) {
    err = memcpy_s(base + aic_que_offset + i * aic_que_size, aic_que_size, skTaskCube.GetTaskQue(), aic_que_size);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s AIC queue%zu failed, ret=%d", i + 1, err);
      return {};
    }
  }

  // Copy AIV queues
  for (size_t i = 0; i < 4; i++) {
    err = memcpy_s(base + aiv_que_offset + i * aiv_que_size, aiv_que_size, skTaskVec.GetTaskQue(), aiv_que_size);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s AIV queue%zu failed, ret=%d", i + 1, err);
      return {};
    }
  }

  if (counter_size > 0) {
    err = memset_s(base + counter_offset, counter_size, 0, counter_size);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memset_s counter failed, ret=%d", static_cast<int>(err));
      return {};
    }
  }
  if (dfx_size > 0 && dfxInfos != nullptr) {
    err = memcpy_s((uint8_t *)args.Get() + args.Get()->skHeader.dfxOffset, dfx_size, dfxInfos,
                   dfxCount * sizeof(SkDfxInfo));
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s dfx failed, ret=%d", static_cast<int>(err));
      return {};
    }
  }

  // 写入事件配置
  SkEventConfig *dstEventConfig = (SkEventConfig *)(base + event_config_offset);
  if (eventConfig != nullptr) {
    err = memcpy_s(dstEventConfig, event_config_size, eventConfig, event_config_size);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s eventConfig failed, ret=%d\n", static_cast<int>(err));
      return DeviceArgsPtr();
    }
  } else {
    // 默认值：禁用事件记录
    memset_s(dstEventConfig, event_config_size, 0, event_config_size);
    dstEventConfig->enabled = 0;
  }

  args.Get()->skHeader.nodeCnt = dfxCount;
  return args;
}

std::pair<int, int> SkTaskBuilder::GetPreFetchCnt(const ResolvedFunctionInfo &resolved) const {
  std::pair<int, int> preFetchCntValue = std::make_pair(resolved.prefetchCnt[0], resolved.prefetchCnt[1]);
  auto preLoadOptions = opts.GetOption(aclskOptionType::PRELOAD_CODE);

  // default: preLoadValue == 1, use func size to prefetch
  uint32_t preLoadValue = 1;

  if (preLoadOptions != nullptr) {
    preLoadValue = preLoadOptions->GetIntValue();
  }

  if (preLoadValue == 0) {
    // use max size to prefetch
    preFetchCntValue.first = 16;  // cube's max icache size is 32K(16 * 2)
    preFetchCntValue.second = 8;  // vec's max icache size is 16K(8 * 2)
  } else if (preLoadValue == 2) {
    // no preload
    preFetchCntValue.first = 0;
    preFetchCntValue.second = 0;
  }
  // preLoadValue == 1: use func size (default from resolved.prefetchCnt)

  return preFetchCntValue;
}

bool SkTaskBuilder::AddSyncTask(SkTask &skTask, size_t nodeIndex, SkCoreSyncType syncType, uint8_t earlyStartConfig,
                                uint32_t skipCoreCount, SkKernelType relatedType) {
  TaskQue *taskQue = skTask.GetTaskQue();
  if (taskQue == nullptr) {
    SK_LOGE("AddSyncTask failed: get writable task queue failed");
    return false;
  }

  auto syncAllOptions = opts.GetOption(aclskOptionType::DEBUG_SYNC_ALL);
  uint32_t debugSyncAll = 0;
  if (syncAllOptions != nullptr) {
    debugSyncAll = syncAllOptions->GetIntValue();
  }

  TaskInfo &taskInfo = taskQue->taskInfos[taskQue->taskCnt];
  taskInfo.index = nodeIndex;
  taskInfo.type = SkTaskType::TYPE_SYNC;
  taskInfo.args = static_cast<uint32_t>(syncType);
  taskInfo.numBlocks = static_cast<uint8_t>(skipCoreCount);
  taskInfo.extraInfo = static_cast<uint64_t>(earlyStartConfig);
  taskInfo.relatedType = relatedType;
  if (opts.EnableDebug() && debugSyncAll == 1) {
    taskInfo.debugOptions |= kDebugOptionDebugSyncAll;
  }
  taskQue->taskCnt++;
  return true;
}

bool SkTaskBuilder::AddEventTask(SkTask &skTask, SuperKernelBaseNode *node, size_t nodeIndex, SkTaskType taskType) {
  TaskQue *taskQue = skTask.GetTaskQue();
  if (taskQue == nullptr) {
    SK_LOGE("AddEventTask failed: get writable task queue failed");
    return false;
  }

  TaskInfo &taskInfo = taskQue->taskInfos[taskQue->taskCnt];
  taskInfo.index = nodeIndex;
  taskInfo.type = taskType;
  const auto &syncInfos = node->GetNodeInfos().syncInfos;
  uint64_t eventValue = syncInfos.memoryValue;
  if (eventValue == std::numeric_limits<uint64_t>::max()) {
    switch (taskType) {
      case SkTaskType::TYPE_EVENT_NOTIFY:
        eventValue = SK_DEFAULT_NOTIFY_VALUE;
        break;
      case SkTaskType::TYPE_EVENT_WAIT:
        eventValue = SK_DEFAULT_WAIT_VALUE;
        break;
      case SkTaskType::TYPE_EVENT_RESET:
        eventValue = SK_DEFAULT_RESET_VALUE;
        break;
      default:
        eventValue = 0;
        break;
    }
  }

  uint32_t waitFlag = syncInfos.memoryWaitFlag;
  if (taskType != SkTaskType::TYPE_EVENT_WAIT || waitFlag == std::numeric_limits<uint32_t>::max()) {
    waitFlag =
        (taskType == SkTaskType::TYPE_EVENT_WAIT) ? static_cast<uint32_t>(SkMemoryWaitFlag::EQ) : SK_DEFAULT_WRITE_FLAG;
  }

  SetEventTaskArgs(taskInfo, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(syncInfos.addrValue)), eventValue,
                   waitFlag);
  taskQue->taskCnt++;
  return true;
}

bool SkTaskBuilder::ProcessCoreFuncSize(SkDfxInfo *dfxInfo, aclrtBinHandle binHdl, const void *binHostAddr,
                                        uint32_t binHostSize, const ResolvedFunctionInfo &resolved, int coreIndex,
                                        int binIndex, const char *coreName) {
  SK_LOGD("ProcessCoreFuncSize: Processing %s (coreIndex=%d), funcAddr=0x%lx, funcOffset=0x%lx", coreName, coreIndex,
          resolved.funcAddr[coreIndex], resolved.funcOffset[coreIndex]);

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;
  bool getInfoRet = GetFuncSymbolInfo(binHdl, static_cast<const char *>(binHostAddr), binHostSize,
                                      resolved.funcOffset[coreIndex], symbolName, funcSize, symbolBind);
  SK_LOGD("ProcessCoreFuncSize: GetFuncSymbolInfo(%s) returned=%d, offset=0x%lx, symbolName=%s, size=0x%lx, bind=%s",
          coreName, getInfoRet, resolved.funcOffset[coreIndex], symbolName.c_str(), funcSize, symbolBind.c_str());

  if (getInfoRet) {
    if (coreIndex == 0) {
      dfxInfo->aicSize = static_cast<uint32_t>(funcSize);
      dfxInfo->entryAic[binIndex] = resolved.funcAddr[0];
      dfxInfo->aicFuncOffset[binIndex] = resolved.funcOffset[0];
      SK_LOGD("ProcessCoreFuncSize: Set aicSize=0x%x, entryAic[%d]=0x%lx, aicFuncOffset[%d]=0x%lx", dfxInfo->aicSize,
              binIndex, dfxInfo->entryAic[binIndex], binIndex, dfxInfo->aicFuncOffset[binIndex]);
    } else if (coreIndex == 1) {
      dfxInfo->aivSize = static_cast<uint32_t>(funcSize);
      dfxInfo->entryAiv[binIndex] = resolved.funcAddr[1];
      dfxInfo->aivFuncOffset[binIndex] = resolved.funcOffset[1];
      SK_LOGD("ProcessCoreFuncSize: Set aivSize=0x%x, entryAiv[%d]=0x%lx, aivFuncOffset[%d]=0x%lx", dfxInfo->aivSize,
              binIndex, dfxInfo->entryAiv[binIndex], binIndex, dfxInfo->aivFuncOffset[binIndex]);
    }
    return true;
  }

  SK_LOGW("ProcessCoreFuncSize: GetFuncSymbolInfo failed for %s", coreName);
  return false;
}

bool SkTaskBuilder::UpdateDfxInfo(SkDfxInfo *dfxInfo, const KernelInfos &kernelInfo,
                                  const ResolvedFunctionInfo &resolved, int binIndex, int addrIndex) {
  if (dfxInfo == nullptr) {
    return true;
  }

  SK_LOGD("UpdateDfxInfo: Processing dfxInfo for binIndex=%d, addrIndex=%d, funcName=%s", binIndex, addrIndex,
          kernelInfo.funcName.c_str());

  dfxInfo->binHdl = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernelInfo.binHdl));
  dfxInfo->funcHdlOri = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernelInfo.funcHdl));
  dfxInfo->numBlocks = kernelInfo.numBlocks;
  dfxInfo->cubeNum = kernelInfo.cubeNum;
  dfxInfo->vecNum = kernelInfo.vecNum;
  SK_LOGD("UpdateDfxInfo: binHdl=0x%lx, funcHdlOri=0x%lx, numBlocks=%u, cubeNum=%u, vecNum=%u", dfxInfo->binHdl,
          dfxInfo->funcHdlOri, dfxInfo->numBlocks, dfxInfo->cubeNum, dfxInfo->vecNum);

  void *binHostAddr = nullptr;
  uint32_t binHostSize = 0;
  int rtRet = rtGetBinBuffer(kernelInfo.binHdl, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize);
  if (rtRet != 0) {
    SK_LOGW("UpdateDfxInfo: Failed to get bin buffer, rtRet=%d, using default size 0", rtRet);
    return true;
  }

  SK_LOGD("UpdateDfxInfo: Successfully got bin buffer, binHostAddr=0x%lx, binHostSize=%u", (uint64_t)binHostAddr,
          binHostSize);

  if (addrIndex == 0 && resolved.funcAddr[0] != 0) {
    ProcessCoreFuncSize(dfxInfo, kernelInfo.binHdl, binHostAddr, binHostSize, resolved, 0, binIndex, "AIC");
  } else {
    SK_LOGD("UpdateDfxInfo: Skipping AIC processing, addrIndex=%d, funcAddr[0]=0x%lx", addrIndex, resolved.funcAddr[0]);
  }

  if (addrIndex == 1 && resolved.funcAddr[1] != 0) {
    ProcessCoreFuncSize(dfxInfo, kernelInfo.binHdl, binHostAddr, binHostSize, resolved, 1, binIndex, "AIV");
  } else {
    SK_LOGD("UpdateDfxInfo: Skipping AIV processing, addrIndex=%d, funcAddr[1]=0x%lx", addrIndex, resolved.funcAddr[1]);
  }

  return true;
}

bool SkTaskBuilder::AddFuncTask(SkTask &skTask, SuperKernelBaseNode *node, SkDfxInfo *dfxInfo, size_t nodeIndex,
                                int addrIndex, int binCount, SkTaskType taskType, uint32_t numBlocks) {
  TaskQue *taskQue = skTask.GetTaskQue();
  if (taskQue == nullptr) {
    SK_LOGE("AddFuncTask failed: get writable task queue failed");
    return false;
  }

  auto disableDcciOptions = opts.GetOption(aclskOptionType::DCCI_DISABLE_ON_KERNEL);
  auto dcciBeforeKernelStartOptions = opts.GetOption(aclskOptionType::DCCI_BEFORE_KERNEL_START);
  auto dcciAfterKernelEndOptions = opts.GetOption(aclskOptionType::DCCI_AFTER_KERNEL_END);
  auto crossCoreSyncCheckOptions = opts.GetOption(aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK);
  uint32_t debugCrossCoreSyncCheck = 0;
  if (crossCoreSyncCheckOptions != nullptr) {
    debugCrossCoreSyncCheck = crossCoreSyncCheckOptions->GetIntValue();
  }
  std::vector<std::string> disableDcciList;
  std::vector<std::string> dcciBeforeKernelStartList;
  std::vector<std::string> dcciAfterKernelEndList;
  if (disableDcciOptions != nullptr) {
    disableDcciList = disableDcciOptions->GetStringListValue();
  }

  if (dcciBeforeKernelStartOptions != nullptr) {
    dcciBeforeKernelStartList = dcciBeforeKernelStartOptions->GetStringListValue();
  }

  if (dcciAfterKernelEndOptions != nullptr) {
    dcciAfterKernelEndList = dcciAfterKernelEndOptions->GetStringListValue();
  }

  if (node->GetNodeType() != SkNodeType::NODE_KERNEL) {
    SK_LOGE("AddFuncTask failed: unsupported node type for KERNEL task");
    return false;
  }

  const KernelInfos &kernelInfo = node->GetNodeInfos().kernelInfos;
  TaskInfo &taskInfo = taskQue->taskInfos[taskQue->taskCnt];
  taskInfo.index = nodeIndex;
  taskInfo.type = taskType;
  taskInfo.relatedType = kernelInfo.kernelType;
  taskInfo.isSimtKernel = (taskType == SkTaskType::TYPE_FUNC && kernelInfo.isSimtOp) ? 1 : 0;
  taskInfo.numBlocks = numBlocks;
  if (kernelInfo.resolvedNum != binCount) {
    SK_LOGW(
        "mismatch num between sub sk registered and sk option, funcName: %s, registered: %d,"
        "option: %d",
        kernelInfo.funcName.c_str(), kernelInfo.resolvedNum, binCount);
  }
  for (int i = 0; i < binCount; i++) {
    const ResolvedFunctionInfo &resolved = kernelInfo.resolvedFuncs[i];
    if (taskType == SkTaskType::TYPE_PRELOAD) {
      std::pair<int, int> prefetchCntValue = GetPreFetchCnt(resolved);
      if (i == 0) {
        SK_LOGI("kernel name: %s, prefetch count: %d %d", kernelInfo.funcName.c_str(), prefetchCntValue.first,
                prefetchCntValue.second);
      }
      taskInfo.args = addrIndex == 0 ? prefetchCntValue.first : prefetchCntValue.second;
      taskInfo.argsSize = static_cast<uint32_t>(node->GetTaskParams().kernelTaskParams.argsSize);
      taskInfo.extraInfo = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernelInfo.devArgs));
    }

    uint64_t addr = resolved.funcAddr[addrIndex];
    if (addr == 0) {
      SK_LOGE("AddFuncTask failed: unresolved function address, nodeIndex=%zu, binIndex=%d, addrIndex=%d", nodeIndex, i,
              addrIndex);
      return false;
    }

    taskInfo.entryCnt++;
    taskInfo.entry[i] = addr;

    if (taskType == SkTaskType::TYPE_FUNC && dfxInfo) {
      UpdateDfxInfo(dfxInfo, kernelInfo, resolved, i, addrIndex);
    }
  }

  if (taskType == SkTaskType::TYPE_FUNC) {
    taskInfo.args = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernelInfo.devArgs));
    taskInfo.extraInfo = nodeIndex < taskSyncInfos_.size()
                             ? static_cast<uint64_t>(taskSyncInfos_[nodeIndex].earlyStartInfo.funcEarlyStartConfig)
                             : static_cast<uint64_t>(SkEarlyStartMask::NONE);
    taskInfo.argsSize = static_cast<uint32_t>(node->GetTaskParams().kernelTaskParams.argsSize);
    skTask.numBlocks = std::max(skTask.numBlocks, static_cast<uint32_t>(taskInfo.numBlocks));
    skTask.funcCnt++;

    taskInfo.debugOptions |= BuildFuncTaskDebugOptions(opts, kernelInfo, disableDcciList, dcciBeforeKernelStartList,
                                                       dcciAfterKernelEndList, debugCrossCoreSyncCheck);
  }
  taskQue->taskCnt++;
  return true;
}

bool SkTaskBuilder::DispatchFuncTask(SkTask &skTaskCube, SkTask &skTaskVec, SuperKernelBaseNode *node,
                                     SkDfxInfo *dfxInfo, size_t nodeIndex, int binCount, SkTaskType taskType,
                                     SkQueueType queueType, const ScopeCoreInfo &scopeCoreInfo) {
  if (node->GetNodeType() != SkNodeType::NODE_KERNEL) {
    SK_LOGE("DispatchFuncTask failed: unsupported node type for FUNC/PRELOAD task");
    return false;
  }
  const uint32_t cubeNum = UsesAic(queueType) ? GetTaskNumBlocks(*node, SkQueueType::AIC, scopeCoreInfo) : 0;
  const uint32_t vectorNum = UsesAiv(queueType) ? GetTaskNumBlocks(*node, SkQueueType::AIV, scopeCoreInfo) : 0;

  switch (queueType) {
    case SkQueueType::AIV: {
      if (!AddFuncTask(skTaskVec, node, dfxInfo, nodeIndex, 1, binCount, taskType, vectorNum)) {
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), node->Format().c_str());
      break;
    }
    case SkQueueType::AIC: {
      if (!AddFuncTask(skTaskCube, node, dfxInfo, nodeIndex, 0, binCount, taskType, cubeNum)) {
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), node->Format().c_str());
      break;
    }
    case SkQueueType::MIX_1_1: {
      if (!AddFuncTask(skTaskCube, node, dfxInfo, nodeIndex, 0, binCount, taskType, cubeNum)) {
        return false;
      }
      if (!AddFuncTask(skTaskVec, node, dfxInfo, nodeIndex, 1, binCount, taskType, vectorNum)) {
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), node->Format().c_str());
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), node->Format().c_str());
      break;
    }
    case SkQueueType::MIX_1_2: {
      if (!AddFuncTask(skTaskCube, node, dfxInfo, nodeIndex, 0, binCount, taskType, cubeNum)) {
        return false;
      }
      if (!AddFuncTask(skTaskVec, node, dfxInfo, nodeIndex, 1, binCount, taskType, vectorNum)) {
        return false;
      }
      skTaskVec.nodeType = SkKernelType::MIX_AIC_1_2;
      skTaskCube.nodeType = SkKernelType::MIX_AIC_1_2;
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), node->Format().c_str());
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), node->Format().c_str());
      break;
    }
    default:
      SK_LOGE("DispatchFuncTask failed: unsupported kernel type %s for super kernel", to_string(queueType));
      return false;
  }
  return true;
}

bool SkTaskBuilder::DispatchEventTask(SkTask &skTaskCube, SkTask &skTaskVec, SuperKernelBaseNode *node,
                                      size_t nodeIndex, SkTaskType taskType, SkQueueType queueType) {
  if (node->GetNodeType() != SkNodeType::NODE_NOTIFY && node->GetNodeType() != SkNodeType::NODE_WAIT &&
      node->GetNodeType() != SkNodeType::NODE_RESET) {
    SK_LOGE("DispatchEventTask failed: unsupported node type for EVENT_NOTIFY/EVENT_WAIT/EVENT_RESET task");
    return false;
  }
  SkTask *targetTask = nullptr;
  std::string targetQue;
  switch (queueType) {
    case SkQueueType::AIC:
      targetTask = &skTaskCube;
      targetQue = "AIC";
      break;
    case SkQueueType::AIV:
      targetTask = &skTaskVec;
      targetQue = "AIV";
      break;
    case SkQueueType::MIX_1_1:
    case SkQueueType::MIX_1_2:
    default:
      targetTask = &skTaskVec;
      targetQue = "AIV";
      break;
  }
  if (!AddEventTask(*targetTask, node, nodeIndex, taskType)) {
    SK_LOGE("DispatchEventTask failed: add event task failed");
    return false;
  }
  SK_LOGI("    task insert: task %zu, [queue=%s], [type=%s], [nodeInfo=%s]", nodeIndex, targetQue.c_str(),
          to_string(taskType), node->Format().c_str());
  return true;
}

// ========== Helper: batch dispatch of sync tasks ==========

bool SkTaskBuilder::DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, size_t nodeIndex,
                                      const std::map<size_t, SyncDirection> &syncInfo, bool isSend,
                                      SkQueueType queueType) {
  for (const auto &kv : syncInfo) {
    size_t peerIdx = kv.first;
    SyncDirection dir = kv.second;
    SkQueueType peerType = taskSyncInfos_[peerIdx].queueType;
    auto syncType = GetSyncTypesForDirection(dir, isSend);

    // Determine dispatch target queues based on sync type.
    bool addToAicQue = false;
    bool addToAivQue = false;
    SkQueueType prevType = SkQueueType::UNKNOWN;
    SkQueueType nextType = SkQueueType::UNKNOWN;
    switch (syncType) {
      case SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC:
        addToAicQue = true;
        addToAivQue = false;
        prevType = queueType;
        nextType = SkQueueType::UNKNOWN;
        break;
      case SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV:
        addToAicQue = false;
        addToAivQue = true;
        prevType = queueType;
        nextType = SkQueueType::UNKNOWN;
        break;
      case SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC:
        // AIV SET signal: insert into AIV queue.
        addToAicQue = false;
        addToAivQue = true;
        prevType = queueType;
        nextType = peerType;
        break;
      case SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC:
        // AIV->AIC WAIT signal: insert into AIC queue.
        addToAicQue = true;
        addToAivQue = false;
        prevType = peerType;
        nextType = queueType;
        break;
      case SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV:
        // AIC SET signal: insert into AIC queue.
        addToAicQue = true;
        addToAivQue = false;
        prevType = queueType;
        nextType = peerType;
        break;
      case SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV:
        // AIC->AIV WAIT signal: insert into AIV queue.
        addToAicQue = false;
        addToAivQue = true;
        prevType = peerType;
        nextType = queueType;
        break;
      case SkCoreSyncType::ALL_SYNC:
        addToAicQue = true;
        addToAivQue = true;
        prevType = SkQueueType::UNKNOWN;
        nextType = SkQueueType::UNKNOWN;
        break;
      default:
        SK_LOGE("unknown sync type %s, skipping", to_string(syncType));
        return false;
    }

    if (addToAicQue) {
      if (!AddSyncTask(skTaskCube, nodeIndex, syncType)) {
        SK_LOGE("DispatchSyncTasks failed: add sync task to aic queue failed");
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [flag=%s], [prev=%s, next=%s]", nodeIndex,
              to_string(SkTaskType::TYPE_SYNC), to_string(syncType), to_string(prevType), to_string(nextType));
    }
    if (addToAivQue) {
      if (!AddSyncTask(skTaskVec, nodeIndex, syncType)) {
        SK_LOGE("DispatchSyncTasks failed: add sync task to aiv queue failed");
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [flag=%s], [prev=%s, next=%s]", nodeIndex,
              to_string(SkTaskType::TYPE_SYNC), to_string(syncType), to_string(prevType), to_string(nextType));
    }
  }
  return true;
}

bool SkTaskBuilder::DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, size_t nodeIndex,
                                      const EarlyStartInfo &earlyStartInfo, bool isSend, SkQueueType queueType,
                                      const ScopeCoreInfo &scopeCoreInfo) {
  std::map<SkEarlyStartMask, SkCoreSyncType> maskToSyncType = {
      {SkEarlyStartMask::AIC_TO_AIC_SET, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC},
      {SkEarlyStartMask::AIC_TO_AIC_WAIT, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC},
      {SkEarlyStartMask::AIV_TO_AIV_SET, SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV},
      {SkEarlyStartMask::AIV_TO_AIV_WAIT, SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV},
      {SkEarlyStartMask::AIC_TO_AIV_SET, SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV},
      {SkEarlyStartMask::AIC_TO_AIV_WAIT, SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV},
      {SkEarlyStartMask::AIV_TO_AIC_SET, SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC},
      {SkEarlyStartMask::AIV_TO_AIC_WAIT, SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC},
  };
  auto addEarlyStartSync = [&](SkTask &skTask, SkEarlyStartMask mask, SuperKernelBaseNode *relatedNode,
                               SkQueueType queueType) -> bool {
    if (!earlyStartInfo.CheckSyncMask(mask)) {
      return true;
    }
    auto it = maskToSyncType.find(mask);
    if (it == maskToSyncType.end()) {
      SK_LOGE("DispatchSyncTasks failed: unknown early start sync mask %s", to_string(mask));
      return false;
    }
    if (relatedNode == nullptr) {
      SK_LOGE("DispatchSyncTasks failed: related node is null");
      return false;
    }
    const uint32_t activeCoreCount = GetTaskNumBlocks(*relatedNode, queueType, scopeCoreInfo);
    const SkKernelType relatedType = GetKernelType(relatedNode);
    if (!AddSyncTask(skTask, nodeIndex, it->second, static_cast<uint8_t>(mask), activeCoreCount, relatedType)) {
      SK_LOGE("DispatchSyncTasks failed: add early start sync task failed, mask=%s", to_string(mask));
      return false;
    }
    SK_LOGI("    task insert: task %zu, [queue=%s], [type=%s], [flag=%s], [activeCoreCount=%u]", nodeIndex,
            to_string(queueType), to_string(SkTaskType::TYPE_SYNC), to_string(it->second), activeCoreCount);
    return true;
  };

  if (!isSend) {
    // add aic task to aic task que
    if (!addEarlyStartSync(skTaskCube, SkEarlyStartMask::AIV_TO_AIC_WAIT, earlyStartInfo.relatedWaitNode,
                           SkQueueType::AIC)) {
      return false;
    }
    // add aiv task to aiv task que
    if (!addEarlyStartSync(skTaskVec, SkEarlyStartMask::AIC_TO_AIV_WAIT, earlyStartInfo.relatedWaitNode,
                           SkQueueType::AIV)) {
      return false;
    }
  } else {
    // add aic task to aic task que
    if (!addEarlyStartSync(skTaskCube, SkEarlyStartMask::AIC_TO_AIC_SET, earlyStartInfo.relatedSetNode,
                           SkQueueType::AIC)) {
      return false;
    }
    if (!addEarlyStartSync(skTaskCube, SkEarlyStartMask::AIC_TO_AIC_WAIT, earlyStartInfo.nextAicRelatedNode,
                           SkQueueType::AIC)) {
      return false;
    }
    if (!addEarlyStartSync(skTaskCube, SkEarlyStartMask::AIC_TO_AIV_SET, earlyStartInfo.nextAicRelatedNode,
                           SkQueueType::AIC)) {
      return false;
    }
    // add aiv task to aiv task que
    if (!addEarlyStartSync(skTaskVec, SkEarlyStartMask::AIV_TO_AIV_SET, earlyStartInfo.relatedSetNode,
                           SkQueueType::AIV)) {
      return false;
    }
    if (!addEarlyStartSync(skTaskVec, SkEarlyStartMask::AIV_TO_AIV_WAIT, earlyStartInfo.nextAivRelatedNode,
                           SkQueueType::AIV)) {
      return false;
    }
    if (!addEarlyStartSync(skTaskVec, SkEarlyStartMask::AIV_TO_AIC_SET, earlyStartInfo.nextAivRelatedNode,
                           SkQueueType::AIV)) {
      return false;
    }
  }
  return true;
}

// ========== 辅助函数：根据配置生成 entry 函数名 ==========
namespace {

// entryFuncName 配置标记
enum class EntryFuncFlag : uint8_t {
  NONE = 0U,
  DEBUG = 1U << 0,
  DUMP_PROFILING = 1U << 1,
  OP_TRACE = 1U << 2,
  EARLY_START = 1U << 3,
};

/**
 * @brief 根据 kernel 类型获取基础 entry 函数名
 */
const char *GetBaseEntryFuncName(SkKernelType kernelType) {
  switch (kernelType) {
    case SkKernelType::AIV_ONLY:
      return "sk_entry_aiv";
    case SkKernelType::AIC_ONLY:
      return "sk_entry_aic";
    case SkKernelType::MIX_AIC_1_1:
      return "sk_entry_mix11";
    case SkKernelType::MIX_AIC_1_2:
      return "sk_entry_mix12";
    default:
      return "sk_entry_aic";
  }
}

/**
 * @brief 根据标记位组合生成最终的 entry 函数名
 */
std::string BuildEntryFuncName(const char *baseName, EntryFuncFlag flags) {
  std::string funcName = baseName;
  uint8_t flagValue = static_cast<uint8_t>(flags);
  if ((flagValue & static_cast<uint8_t>(EntryFuncFlag::DEBUG)) != 0) {
    funcName += "_debug";
  }
  if ((flagValue & static_cast<uint8_t>(EntryFuncFlag::DUMP_PROFILING)) != 0) {
    funcName += "_dump_profiling";
  }
  if ((flagValue & static_cast<uint8_t>(EntryFuncFlag::OP_TRACE)) != 0) {
    funcName += "_op_trace";
  }
  if ((flagValue & static_cast<uint8_t>(EntryFuncFlag::EARLY_START)) != 0) {
    funcName += "_early_start";
  }
  return funcName;
}

}  // namespace

bool SkTaskBuilder::CalculateSkScheMode(const std::vector<SuperKernelBaseNode *> &tasks, const SkTask &aicTask,
                                        const SkTask &aivTask) const {
  for (const SkTask *skTask : {&aicTask, &aivTask}) {
    const TaskQue *queue = skTask->GetTaskQue();
    if (queue == nullptr) {
      continue;
    }
    for (uint32_t i = 0; i < queue->taskCnt; ++i) {
      const TaskInfo &taskInfo = queue->taskInfos[i];
      if (taskInfo.type == SkTaskType::TYPE_FUNC) {
        const uint64_t nodeId = GetTaskNodeIdByIndex(tasks, taskInfo.index);
        const auto *node = graph_.GetNodeById(nodeId);
        if (node->IsScheModeOn()) {
          return true;
        }

        // DEBUG_CROSS_CORE_SYNC_CHECK calls SyncAll after MIX sub-kernels inside FUNC, without a SYNC task.
        const bool enableCrossCoreSyncCheck = (taskInfo.debugOptions & kDebugOptionCrossCoreSyncCheck) != 0;
        const bool isMixKernel =
            taskInfo.relatedType == SkKernelType::MIX_AIC_1_1 || taskInfo.relatedType == SkKernelType::MIX_AIC_1_2;
        if (enableCrossCoreSyncCheck && isMixKernel) {
          return true;
        }
      } else if (taskInfo.type == SkTaskType::TYPE_SYNC) {
        // Final sync types already cover early-start (C-C/V-V) and DEBUG_SYNC_ALL (ALL_SYNC).
        // Either all C cores or all V cores needing to synchronize requires batch scheduling.
        const auto syncType = static_cast<SkCoreSyncType>(taskInfo.args);
        if (syncType == SkCoreSyncType::ALL_SYNC || syncType == SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC ||
            syncType == SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV) {
          return true;
        }
      }
    }
  }
  return false;
}

SkHostEntryInfo SkTaskBuilder::GenEntryInfo(SkTask &skTaskCube, SkTask &skTaskVec, const ScopeCoreInfo &scopeCoreInfo,
                                            bool useSimtEntry) {
  SkHostEntryInfo entryInfo;
  bool enableDebug = opts.EnableDebug();
  // ASCEND_PROF_SK_ON enables profiling.
  const char *profilingEnv = std::getenv("ASCEND_PROF_SK_ON");
  bool enableProfiling = (profilingEnv != nullptr && std::string(profilingEnv) != "0");
  // DEBUG_OP_EXEC_TRACE enables op trace.
  auto opExecTraceOpt = opts.GetOption(aclskOptionType::DEBUG_OP_EXEC_TRACE);
  bool enableOpTrace = (opExecTraceOpt != nullptr && opExecTraceOpt->GetIntValue() == 1);
  // Cross-core sync checking requires an op-trace entry kernel.
  auto crossCoreSyncCheckOpt = opts.GetOption(aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK);
  if (crossCoreSyncCheckOpt != nullptr && crossCoreSyncCheckOpt->GetIntValue() == 1) {
    enableOpTrace = true;
  }
  // EARLY_START launches the SK early to reduce first-round scheduling latency.
  auto earlyStartOpt = opts.GetOption(aclskOptionType::EARLY_START);
  bool enableEarlyStart = (earlyStartOpt != nullptr && earlyStartOpt->GetIntValue() == 1);

  SK_LOGI(
      "GenEntryInfo: enableDebug=%d, enableProfiling=%d, enableOpTrace=%d, enableEarlyStart=%d, "
      "useSimtEntry=%d",
      enableDebug, enableProfiling, enableOpTrace, enableEarlyStart, useSimtEntry);

  // Scope splitting finalizes the SK core result; TaskBuilder only consumes it.
  entryInfo.entryType = scopeCoreInfo.type;
  entryInfo.numBlocks = scopeCoreInfo.numBlocks;
  skTaskCube.numBlocks = scopeCoreInfo.GetCubeNum();
  skTaskVec.numBlocks = scopeCoreInfo.GetVectorNum();

  // Build the entry function name from the enabled runtime and debug features.
  uint8_t flags = static_cast<uint8_t>(EntryFuncFlag::NONE);
  if (enableDebug) {
    flags = flags | static_cast<uint8_t>(EntryFuncFlag::DEBUG);
  }
  if (enableProfiling) {
    flags = flags | static_cast<uint8_t>(EntryFuncFlag::DUMP_PROFILING);
  }
  if (enableOpTrace) {
    flags = flags | static_cast<uint8_t>(EntryFuncFlag::OP_TRACE);
  }
  if (enableEarlyStart) {
    flags = flags | static_cast<uint8_t>(EntryFuncFlag::EARLY_START);
  }
  EntryFuncFlag funcFlags = static_cast<EntryFuncFlag>(flags);

  const char *baseName = GetBaseEntryFuncName(entryInfo.entryType);
  std::string entryFuncName = BuildEntryFuncName(baseName, funcFlags);
  if (useSimtEntry) {
    entryFuncName += "_simt";
  }
  entryInfo.skEntryFunc = ResolveSkEntryFunc(entryFuncName.c_str());
  if (entryInfo.skEntryFunc == nullptr) {
    SK_LOGE("failed to resolve sk entry function: entryFuncName=%s", entryFuncName.c_str());
    return {};
  }
  SK_LOGI("sk entry resolved: type=%s, funcName=%s, funcHandle=%p, numBlocks=%d", to_string(entryInfo.entryType),
          entryFuncName.c_str(), entryInfo.skEntryFunc, entryInfo.numBlocks);
  return entryInfo;
}

// generate the final launch info for super kernel execution
SkBuildResult SkTaskBuilder::Build(std::string skFuncName, const std::vector<SuperKernelBaseNode *> &tasks,
                                   const std::vector<SuperKernelBaseNode *> &customTasks, uint16_t scopeId,
                                   const ScopeCoreInfo &scopeCoreInfo) {
  // Post-process should already guarantee non-empty tasks.
  if (tasks.empty()) {
    SK_LOGE("Build failed: no task to build for super kernel");
    return {};
  }
  if (!scopeCoreInfo.IsValid()) {
    SK_LOGE("Build failed: scope %u has no final SK core result", scopeId);
    return {};
  }
  for (const auto *task : tasks) {
    if (task->GetNodeType() != SkNodeType::NODE_KERNEL || ValidateTaskCoreRequirement(*task, scopeCoreInfo)) {
      continue;
    }
    SK_LOGE(
        "kernel %lu core requirement does not match final SK result, taskCore={cube:%u,vec:%u}, "
        "scheMode=%d, blockDimScaleUp=%d, skCore={cube:%u,vec:%u}, skType=%s",
        task->GetNodeId(), task->GetCubeNum(), task->GetVecNum(), task->IsScheModeOn(),
        task->GetNodeInfos().kernelInfos.capBits.blockDimScaleUp, scopeCoreInfo.GetCubeNum(),
        scopeCoreInfo.GetVectorNum(), to_string(scopeCoreInfo.type));
    return {};
  }

  const size_t taskCount = tasks.size();

  size_t cap = taskCount * (uint8_t)SkTaskType::TYPE_MAX;
  SkTask aicTask;
  SkTask aivTask;
  if (!aicTask.Init(cap)) {
    SK_LOGE("Build failed: init aic task queue failed, cap=%zu", cap);
    return {};
  }
  if (!aivTask.Init(cap)) {
    SK_LOGE("Build failed: init aiv task queue failed, cap=%zu", cap);
    return {};
  }

  const auto *debugSyncOpt = opts.GetOption(aclskOptionType::DEBUG_SYNC_ALL);
  const bool debugSyncAll = (debugSyncOpt != nullptr && debugSyncOpt->GetIntValue() != 0);

  if (taskCount > (std::numeric_limits<size_t>::max() / sizeof(SkDfxInfo))) {
    SK_LOGE("invalid dfxInfos alloc size, taskCount=%zu", taskCount);
    return {};
  }

  size_t dfxBytes = taskCount * sizeof(SkDfxInfo);
  std::unique_ptr<uint8_t[]> dfxStorage = std::make_unique<uint8_t[]>(dfxBytes);
  SkDfxInfo *dfxInfos = reinterpret_cast<SkDfxInfo *>(dfxStorage.get());
  errno_t initErr = memset_s(dfxInfos, dfxBytes, 0, dfxBytes);
  if (initErr != 0) {
    SK_LOGE("init dfxInfos failed, ret=%d", static_cast<int>(initErr));
    return {};
  }

  int splitBinCount = 4;
  auto splitOptions = opts.GetOption(aclskOptionType::SPLIT_MODE);
  if (splitOptions != nullptr) {
    splitBinCount = splitOptions->GetIntValue();
  }

  if (opts.IsInnerOptionEnabled(SkInnerOptionType::MIX_KERNEL_SPLIT)) {
    if (!PrecomputeSyncRelationsByMixGroups(tasks)) {
      SK_LOGE("Build failed: precompute sync relations with mix kernel split failed");
      return {};
    }
  } else {
    // ========== precompute sync relations from graph topology ==========
    SK_LOGI("Build phase-1 begin: precompute sync relations, taskCount=%zu", taskCount);
    if (!PrecomputeSyncRelationsFromGraph(tasks)) {
      SK_LOGE("Build failed: precompute sync relations failed");
      return {};
    }

    // ========== optimize sync relations ==========
    SK_LOGI("Build phase-2 begin: optimize sync relations");
    OptimizeSyncRelations(tasks);
  }

  // ========== Phase 2: apply early-start related optimization passes if enabled ==========
  const auto *earlyStartOpt = opts.GetOption(aclskOptionType::EARLY_START);
  const bool enableEarlyStart =
      (earlyStartOpt != nullptr && earlyStartOpt->GetIntValue() == aclskEarlyStartValue::ACLSK_EARLY_START_ENABLED);
  if (enableEarlyStart) {
    if (!ApplyEarlyStartSyncPass(tasks)) {
      SK_LOGE("Build failed: apply early-start sync pass failed");
      return {};
    }
  }

  // ========== Phase 3: attach sync metadata for custom tasks ==========
  if (!customTasks.empty()) {
    SK_LOGI("add sync info for custom tasks, customTaskCount=%zu", customTasks.size());
    taskSyncInfos_.back().crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::ALL_SYNC;
  }

  // In debug mode, force all tasks into ALL_SYNC for traceability.
  if (opts.EnableDebug() && debugSyncAll) {
    SK_LOGI("tracing sync all is enabled, now clear all sync info, and only left cross sync info. task count is %d",
            taskCount);
    for (int i = 0; i < static_cast<int>(taskCount); i++) {
      // Clear all sync metadata and force ALL_SYNC in debug mode.
      taskSyncInfos_[i].vecRecvInfo.clear();
      taskSyncInfos_[i].cubRecvInfo.clear();
      taskSyncInfos_[i].vecSendInfo.clear();
      taskSyncInfos_[i].cubSendInfo.clear();
      taskSyncInfos_[i].crossSyncInfo.clear();
      taskSyncInfos_[i].earlyStartInfo = {};
      taskSyncInfos_[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::ALL_SYNC;
    }
  }
  // ========== Phase 4: construct final task queues ==========
  SK_LOGI("start build tasks for super kernel, taskCount=%zu", taskCount);
  int preloadCount = 1;
  SK_LOGI("add tasks preload, preload count is %d, task count is %d", preloadCount, taskCount);
  for (int i = 0; i < preloadCount && i < static_cast<int>(taskCount); i++) {
    if (tasks[i]->GetNodeType() == SkNodeType::NODE_KERNEL) {
      SkQueueType queueType = taskSyncInfos_[i].queueType;
      if (!DispatchFuncTask(aicTask, aivTask, tasks[i], dfxInfos + i, i, splitBinCount, SkTaskType::TYPE_PRELOAD,
                            queueType, scopeCoreInfo)) {
        SK_LOGE("Build failed: preload dispatch failed at task index %d", i);
        return {};
      }
    }
  }

  SK_LOGI("start dispatch tasks...");
  for (int i = 0; i < static_cast<int>(taskCount); i++) {
    SK_LOGI("index=%d, nodeType=%s, nodeInfo=%s", i, to_string(tasks[i]->GetNodeType()), tasks[i]->Format().c_str());

    auto &info = taskSyncInfos_[i];
    SkQueueType queueType = info.queueType;

    // 0. Insert early-start WAIT sync if applicable.
    if (enableEarlyStart) {
      if (!DispatchSyncTasks(aicTask, aivTask, i, info.earlyStartInfo, false, queueType, scopeCoreInfo)) {
        return {};
      }
    }

    // 1. Insert intra-kernel WAIT sync before task execution.
    SK_LOGI("add task sync for vec recv, vec recv size %zu", info.vecRecvInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, i, info.vecRecvInfo, false, queueType)) {
      SK_LOGE("Build failed: dispatch vec recv sync failed at task index %d", i);
      return {};
    }
    SK_LOGI("add task sync for cub recv, cub recv size %zu", info.cubRecvInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, i, info.cubRecvInfo, false, queueType)) {
      SK_LOGE("Build failed: dispatch cub recv sync failed at task index %d", i);
      return {};
    }

    // 2. Add function/event task.
    switch (tasks[i]->GetNodeType()) {
      case SkNodeType::NODE_KERNEL:
        SK_LOGI("add task func, task index is %d", i);
        if (!DispatchFuncTask(aicTask, aivTask, tasks[i], dfxInfos + i, i, splitBinCount, SkTaskType::TYPE_FUNC,
                              queueType, scopeCoreInfo)) {
          SK_LOGE("Build failed: function dispatch failed at task index %d", i);
          return {};
        }
        break;
      case SkNodeType::NODE_NOTIFY:
        SK_LOGI("add task notify, task index is %d", i);
        if (!DispatchEventTask(aicTask, aivTask, tasks[i], i, SkTaskType::TYPE_EVENT_NOTIFY, queueType)) {
          SK_LOGE("Build failed: notify dispatch failed at task index %d", i);
          return {};
        }
        break;
      case SkNodeType::NODE_WAIT:
        SK_LOGI("add task wait, task index is %d", i);
        if (!DispatchEventTask(aicTask, aivTask, tasks[i], i, SkTaskType::TYPE_EVENT_WAIT, queueType)) {
          SK_LOGE("Build failed: wait dispatch failed at task index %d", i);
          return {};
        }
        break;
      case SkNodeType::NODE_RESET:
        SK_LOGI("add task reset, task index is %d", i);
        if (!DispatchEventTask(aicTask, aivTask, tasks[i], i, SkTaskType::TYPE_EVENT_RESET, queueType)) {
          SK_LOGE("Build failed: reset dispatch failed at task index %d", i);
          return {};
        }
        break;
      default:
        SK_LOGE("process task %d: unsupported node type %s, skipping (supported types: KERNEL, NOTIFY, WAIT, RESET)", i,
                to_string(tasks[i]->GetNodeType()));
        return {};
    }

    // 3. Add preload task.
    if (preloadCount > 0 && i + preloadCount < static_cast<int>(taskCount) &&
        tasks[i + preloadCount]->GetNodeType() == SkNodeType::NODE_KERNEL) {
      SkQueueType preloadQueueType = taskSyncInfos_[i + preloadCount].queueType;
      SK_LOGI("add tasks preload, task index is %d", i + preloadCount);
      if (!DispatchFuncTask(aicTask, aivTask, tasks[i + preloadCount], dfxInfos + i + preloadCount, i + preloadCount,
                            splitBinCount, SkTaskType::TYPE_PRELOAD, preloadQueueType, scopeCoreInfo)) {
        SK_LOGE("Build failed: rolling preload dispatch failed at task index %d", i + preloadCount);
        return {};
      }
    }

    // 4. Insert early-start SET sync if applicable.
    if (enableEarlyStart) {
      if (!DispatchSyncTasks(aicTask, aivTask, i, info.earlyStartInfo, true, queueType, scopeCoreInfo)) {
        return {};
      }
    }

    // 5. Insert cross-core sync (or forced debug sync).
    SK_LOGI("add sync task for cross sync info, cross sync info size %zu", info.crossSyncInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, i, info.crossSyncInfo, true, queueType)) {
      SK_LOGE("Build failed: dispatch cross sync failed at task index %d", i);
      return {};
    }

    // 6. Insert intra-kernel SET sync after task execution.
    SK_LOGI("add sync task for vec send, vec send size %zu", info.vecSendInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, i, info.vecSendInfo, true, queueType)) {
      SK_LOGE("Build failed: dispatch vec send sync failed at task index %d", i);
      return {};
    }
    SK_LOGI("add sync task for cub send, cub send size %zu", info.cubSendInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, i, info.cubSendInfo, true, queueType)) {
      SK_LOGE("Build failed: dispatch cub send sync failed at task index %d", i);
      return {};
    }
  }
  SK_LOGI("Finish build tasks for super kernel: aicTaskCnt=%u, aivTaskCnt=%u", aicTask.GetTaskQue()->taskCnt,
          aivTask.GetTaskQue()->taskCnt);

  if (!customTasks.empty()) {
    SK_LOGI("start process custom tasks");
    SK_LOGI("direct add custom tasks");
    constexpr size_t kInvalidCustomTaskIndex = static_cast<size_t>(std::numeric_limits<uint32_t>::max());
    std::unordered_map<uint64_t, const SuperKernelBaseNode *> kernelNodeCache;
    SkQueueType firstKernelEventQueueType = InferFirstKernelEventQueueType(tasks, aicAvailable_, aivAvailable_);
    for (size_t i = 0; i < customTasks.size(); i++) {
      auto *node = customTasks[i];
      // Synthesized event nodes do not belong to a physical graph stream.
      // Keep the existing low-impact convention: reuse the scope-tail anchor
      // stored in preNodeId and infer queue type from nearest previous kernel.
      SkQueueType queueType;
      auto *kernel = FindKernelNodeInDirection(node->GetPreNodeId(), graph_, SearchDirection::PREV, kernelNodeCache);
      if (kernel == nullptr) {
        if (firstKernelEventQueueType == SkQueueType::UNKNOWN) {
          SK_LOGE(
              "Build failed: custom task %zu failed to resolve previous KERNEL node and fallback queue type is "
              "UNKNOWN, nodeId=%lu",
              i, node->GetNodeId());
          return {};
        }
        queueType = firstKernelEventQueueType;
        SK_LOGI("custom task %zu: unable to resolve previous KERNEL node, nodeId=%lu, fallback = %s", i,
                node->GetNodeId(), to_string(queueType));
      } else {
        queueType = ToEventQueueType(ToQueueType(GetKernelInfos(kernel).kernelType), aicAvailable_, aivAvailable_);
      }

      // Custom tasks are event nodes (notify/wait/reset), use EventTask dispatch.
      if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
        if (!DispatchEventTask(aicTask, aivTask, node, kInvalidCustomTaskIndex, SkTaskType::TYPE_EVENT_NOTIFY,
                               queueType)) {
          SK_LOGE("Build failed: custom task %zu notify dispatch failed, nodeId=%lu", i, node->GetNodeId());
          return {};
        }
      } else if (node->GetNodeType() == SkNodeType::NODE_WAIT) {
        if (!DispatchEventTask(aicTask, aivTask, node, kInvalidCustomTaskIndex, SkTaskType::TYPE_EVENT_WAIT,
                               queueType)) {
          SK_LOGE("Build failed: custom task %zu wait dispatch failed, nodeId=%lu", i, node->GetNodeId());
          return {};
        }
      } else if (node->GetNodeType() == SkNodeType::NODE_RESET) {
        if (!DispatchEventTask(aicTask, aivTask, node, kInvalidCustomTaskIndex, SkTaskType::TYPE_EVENT_RESET,
                               queueType)) {
          SK_LOGE("Build failed: custom task %zu reset dispatch failed, nodeId=%lu", i, node->GetNodeId());
          return {};
        }
      } else {
        SK_LOGE("Build failed: custom task %zu has unsupported node type %s, nodeId=%lu", i,
                to_string(node->GetNodeType()), node->GetNodeId());
        return {};
      }
    }
    SK_LOGI("finish process custom tasks");
  }

  bool useSimtEntry = false;
  size_t skMaxDcacheSize = 0;
  if (opts.IsInnerOptionEnabled(SkInnerOptionType::SIMT_OP_SUPPORT)) {
    const SimtDcacheSizeResult dcacheSizeResult = CalculateSimtDcacheSize(tasks);
    if (!dcacheSizeResult.isValid) {
      SK_LOGE("Build failed: SIMT ubuf size validation failed");
      return {};
    }
    useSimtEntry = dcacheSizeResult.hasSimtTask;
    skMaxDcacheSize = dcacheSizeResult.skMaxDcacheSize;
  }

  SK_LOGI("Get entry info...");
  SkHostEntryInfo entryInfo = GenEntryInfo(aicTask, aivTask, scopeCoreInfo, useSimtEntry);
  if (entryInfo.skEntryFunc == nullptr) {
    SK_LOGE("Build failed: GenEntryInfo failed");
    return {};
  }
  entryInfo.nodeCnt = static_cast<uint32_t>(taskCount);

  SK_LOGI("Get entry args...");
  DeviceArgsPtr devArgs = GenEntryArgs(aicTask, aivTask, dfxInfos, static_cast<uint32_t>(taskCount));
  if (devArgs.Get() == nullptr) {
    SK_LOGE("Build failed: GenEntryArgs failed");
    return {};
  }

  DumpDeviceArgs(skFuncName, devArgs.Get(), tasks);

  SkLaunchInfo launchInfo;
  launchInfo.entryInfo = std::move(entryInfo);
  launchInfo.devArgs = std::move(devArgs);
  launchInfo.skFuncName = skFuncName;
  launchInfo.useSimtEntry = useSimtEntry;
  launchInfo.skMaxDcacheSize = skMaxDcacheSize;
  launchInfo.isScheModeOn = CalculateSkScheMode(tasks, aicTask, aivTask);

  // Generate task queue JSON for aggregation
  SK_LOGI("SkTaskToQueueJson: generating task queue JSON for scopeId=%u", scopeId);
  Json taskQueueJson = SkTaskToQueueJson(aicTask, aivTask, scopeId);

  SkBuildResult result;
  result.launchInfo = std::move(launchInfo);
  result.taskQueueJson = std::move(taskQueueJson);

  return result;
}
