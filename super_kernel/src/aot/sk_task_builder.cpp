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

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "runtime/kernel.h"
#include "runtime/rt_external_kernel.h"
#include "securec.h"
#include "sk_log.h"

extern "C" aclrtBinHandle AscendGetEntryBinHandle();
Json SkTaskToQueueJson(const SkTask &aicTask, const SkTask &aivTask, uint16_t scopeId);

namespace {
constexpr size_t kCounterAlignBytes = 64;
constexpr uint32_t kMaxSplitBinCount = 4;
constexpr uint64_t kDebugOptionDebugSyncAll = 0x2;
constexpr uint64_t kDebugOptionCrossCoreSyncCheck = 0x10;

const char *to_string(SyncDirection direction) {
  switch (direction) {
    case SyncDirection::NONE:
      return "NONE";
    case SyncDirection::CUB_TO_CUB:
      return "CUB_TO_CUB";
    case SyncDirection::VEC_TO_VEC:
      return "VEC_TO_VEC";
    case SyncDirection::CUB_TO_VEC:
      return "CUB_TO_VEC";
    case SyncDirection::VEC_TO_CUB:
      return "VEC_TO_CUB";
    case SyncDirection::MIX_TO_MIX:
      return "MIX_TO_MIX";
    case SyncDirection::ALL_SYNC:
      return "ALL_SYNC";
    default:
      return "UNKNOWN";
  }
}
bool UsesAic(SkQueueType type) {
  return type == SkQueueType::AIC || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

const char *to_string(SkTaskKind kind) {
  switch (kind) {
    case SkTaskKind::KERNEL:
      return "KERNEL";
    case SkTaskKind::NOTIFY:
      return "NOTIFY";
    case SkTaskKind::WAIT:
      return "WAIT";
    case SkTaskKind::RESET:
      return "RESET";
    default:
      return "UNKNOWN";
  }
}

bool UsesAiv(SkQueueType type) {
  return type == SkQueueType::AIV || type == SkQueueType::MIX_1_1 || type == SkQueueType::MIX_1_2;
}

const SkKernelDesc *GetKernel(const SkTaskDesc &task) {
  return std::get_if<SkKernelDesc>(&task.payload);
}

const SkEventDesc *GetEvent(const SkTaskDesc &task) {
  return std::get_if<SkEventDesc>(&task.payload);
}

uint64_t GetTaskNodeIdByIndex(const std::vector<SkTaskDesc> &tasks, uint32_t taskIndex) {
  return taskIndex < tasks.size() ? tasks[taskIndex].nodeId : INVALID_TASK_ID;
}

void DumpTaskQueDetail(const TaskQue *que, const char *name, const std::vector<SkTaskDesc> &tasks) {
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
                          const std::vector<SkTaskDesc> &tasks) {
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

void DumpDeviceArgs(std::string skFuncName, const SkDeviceEntryArgs *args, const std::vector<SkTaskDesc> &tasks) {
  {
    SK_LOG_CONTEXT_SIMPLE("sk_device_args.log");
    DumpDeviceArgsDetail(skFuncName, args, tasks);
  }
  DumpDeviceArgsDetail(skFuncName, args, tasks);
}

SkQueueType ExpectedKernelQueue(SkKernelType type) {
  switch (type) {
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
      return SkQueueType::AIC;
  }
}

bool CheckedAdd(size_t left, size_t right, size_t &result) {
  if (left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool CheckedMultiply(size_t left, size_t right, size_t &result) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

bool CheckedAlignUp(size_t value, size_t align, size_t &result) {
  if (align == 0) {
    result = value;
    return true;
  }
  size_t adjusted = 0;
  return CheckedAdd(value, align - 1, adjusted) && CheckedMultiply(adjusted / align, align, result);
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

bool HasAllSyncBarrier(const TaskSyncInfo &info) {
  return std::any_of(info.crossSyncInfo.cbegin(), info.crossSyncInfo.cend(),
                     [](const auto &sync) { return sync.second == SyncDirection::ALL_SYNC; });
}

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

uint32_t GetTaskNumBlocks(const SkKernelDesc &kernel, SkQueueType queue, const SkCoreSpec &coreSpec) {
  const bool vector = queue == SkQueueType::AIV;
  if (GetBoolAttribute(kernel.attributes, SkTaskAttributeId::KERNEL_BLOCK_DIM_SCALE_UP, false)) {
    return vector ? coreSpec.vectorNum : coreSpec.cubeNum;
  }
  const bool adaptMix12 =
      vector && coreSpec.type == SkKernelType::MIX_AIC_1_2 &&
      (kernel.kernelType == SkKernelType::MIX_AIC_1_1 || kernel.kernelType == SkKernelType::MIX_AIC_1_2);
  return adaptMix12 ? kernel.originalNumBlocks * 2U : kernel.originalNumBlocks;
}

enum class EntryFuncFlag : uint8_t {
  NONE = 0U,
  DEBUG = 1U << 0,
  DUMP_PROFILING = 1U << 1,
  OP_TRACE = 1U << 2,
  EARLY_START = 1U << 3,
};

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

bool IsValidTaskKind(SkTaskKind kind) {
  switch (kind) {
    case SkTaskKind::KERNEL:
    case SkTaskKind::NOTIFY:
    case SkTaskKind::WAIT:
    case SkTaskKind::RESET:
      return true;
    default:
      return false;
  }
}

SkTaskType EventTaskType(SkTaskKind kind) {
  switch (kind) {
    case SkTaskKind::NOTIFY:
      return SkTaskType::TYPE_EVENT_NOTIFY;
    case SkTaskKind::WAIT:
      return SkTaskType::TYPE_EVENT_WAIT;
    case SkTaskKind::RESET:
      return SkTaskType::TYPE_EVENT_RESET;
    default:
      return SkTaskType::TYPE_MAX;
  }
}
}  // namespace

void SkTaskBuilder::InsertSyncEvent(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context, size_t preIdx,
                                    size_t currIdx) const {
  SkQueueType preType = tasks[preIdx].queueType;
  SkQueueType currType = tasks[currIdx].queueType;
  // Generate sync direction between producer and consumer.
  SyncDirection dir = GenSyncDirection(preType, currType);

  SK_LOGI("  InsertSyncEvent: task[%zu](%s)(nodeId=%lu) -> task[%zu](%s)(nodeId=%lu), dir=%s", preIdx,
          to_string(preType), context.indexToNodeId[preIdx], currIdx, to_string(currType),
          context.indexToNodeId[currIdx], to_string(dir));

  // Sender side (preIdx): record send relation in the corresponding queue metadata.
  // CUB_TO_VEC: send SET in cub queue, receive WAIT in vec queue.
  // VEC_TO_CUB: send SET in vec queue, receive WAIT in cub queue.
  // MIX_TO_MIX: both directions are required.
  if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::MIX_TO_MIX) {
    context.syncInfos[preIdx].cubSendInfo[currIdx] = SyncDirection::CUB_TO_VEC;
  }
  if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::MIX_TO_MIX) {
    context.syncInfos[preIdx].vecSendInfo[currIdx] = SyncDirection::VEC_TO_CUB;
  }

  // Receiver side (currIdx): record recv relation in the corresponding queue metadata.
  if (dir == SyncDirection::CUB_TO_VEC || dir == SyncDirection::MIX_TO_MIX) {
    context.syncInfos[currIdx].vecRecvInfo[preIdx] = SyncDirection::CUB_TO_VEC;
  }
  if (dir == SyncDirection::VEC_TO_CUB || dir == SyncDirection::MIX_TO_MIX) {
    context.syncInfos[currIdx].cubRecvInfo[preIdx] = SyncDirection::VEC_TO_CUB;
  }
}

void SkTaskBuilder::PrintSyncInfo(const char *stage, const std::vector<SkTaskDesc> &tasks,
                                  const SkTaskContext &context) const {
  SK_LOGI("%s", stage);
  SK_LOGI("[VEC LIST OP]:");
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];
    if (UsesAiv(tasks[i].queueType)) {
      std::string vecSendStr = "vecSend={";
      for (auto &kv : info.vecSendInfo) {
        vecSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      vecSendStr += "}, vecRecv={";
      for (auto &kv : info.vecRecvInfo) {
        vecSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      vecSendStr += "}";
      SK_LOGI("  task[%zu](%s): %s", i, to_string(tasks[i].queueType), vecSendStr.c_str());
    }
  }
  SK_LOGI("[CUB LIST OP]:");
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];
    if (UsesAic(tasks[i].queueType)) {
      std::string cubSendStr = "cubSend={";
      for (auto &kv : info.cubSendInfo) {
        cubSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      cubSendStr += "}, cubRecv={";
      for (auto &kv : info.cubRecvInfo) {
        cubSendStr += std::to_string(kv.first) + ":" + to_string(kv.second) + " ";
      }
      cubSendStr += "}";
      SK_LOGI("  task[%zu](%s): %s", i, to_string(tasks[i].queueType), cubSendStr.c_str());
    }
  }
}

bool SkTaskBuilder::InitTaskSyncInfos(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context) const {
  context.syncInfos.assign(tasks.size(), {});
  context.nodeIdToIndex.clear();
  context.indexToNodeId.resize(tasks.size());
  size_t kernelCount = 0;
  size_t notifyCount = 0;
  size_t waitCount = 0;
  size_t resetCount = 0;
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto &task = tasks[i];
    if (task.queueType == SkQueueType::UNKNOWN) {
      SK_LOGE("queue type was not prepared by optimizer, nodeId=%lu", task.nodeId);
      return false;
    }
    switch (task.kind) {
      case SkTaskKind::KERNEL:
        ++kernelCount;
        break;
      case SkTaskKind::NOTIFY:
        ++notifyCount;
        break;
      case SkTaskKind::WAIT:
        ++waitCount;
        break;
      case SkTaskKind::RESET:
        ++resetCount;
        break;
      default:
        SK_LOGE("unsupported node type for sync info initialization");
        return false;
    }
    if (task.nodeId == INVALID_TASK_ID) {
      SK_LOGE("BuildTaskIndex failed: invalid node ID at index=%zu", i);
      return false;
    }
    if (!context.nodeIdToIndex.emplace(task.nodeId, i).second) {
      SK_LOGE("BuildTaskIndex failed: duplicate nodeId=%lu, index=%zu", task.nodeId, i);
      return false;
    }
    context.indexToNodeId[i] = task.nodeId;
    if (i + 1 == tasks.size() || task.kind == SkTaskKind::NOTIFY || task.kind == SkTaskKind::RESET) {
      continue;
    }
    if (!UsesAic(task.queueType) && !UsesAiv(task.queueType)) {
      SK_LOGE("unsupported kernel type : %s for inter-sync.", to_string(task.queueType));
      return false;
    }
    if (UsesAic(task.queueType)) {
      context.syncInfos[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::CUB_TO_CUB;
    }
    if (UsesAiv(task.queueType)) {
      context.syncInfos[i].crossSyncInfo[static_cast<size_t>(SkQueueType::AIV)] = SyncDirection::VEC_TO_VEC;
    }
  }
  SK_LOGI("Initialized TaskSyncInfos for %zu tasks (%zu KERNEL, %zu NOTIFY, %zu WAIT, %zu RESET)", tasks.size(),
          kernelCount, notifyCount, waitCount, resetCount);
  return true;
}

bool SkTaskBuilder::ExtractSyncRelationsFromGraph(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context,
                                                  bool allowExternalDependencies) const {
  if (context.nodeIdToIndex.size() != tasks.size() || context.indexToNodeId.size() != tasks.size()) {
    SK_LOGE("ExtractSyncRelationsFromGraph failed: task index is not built for taskCount=%zu", tasks.size());
    return false;
  }
  std::vector<std::pair<size_t, size_t>> graphEdges;
  for (size_t i = 0; i < tasks.size(); ++i) {
    for (uint64_t successorId : tasks[i].successorIds) {
      const auto successor = context.nodeIdToIndex.find(successorId);
      if (successor == context.nodeIdToIndex.end()) {
        if (!allowExternalDependencies) {
          SK_LOGE("ExtractSyncRelationsFromGraph failed: missing target nodeId=%lu from source nodeId=%lu", successorId,
                  tasks[i].nodeId);
          return false;
        }
        continue;
      }
      graphEdges.emplace_back(i, successor->second);
    }
  }
  std::sort(graphEdges.begin(), graphEdges.end());
  graphEdges.erase(std::unique(graphEdges.begin(), graphEdges.end()), graphEdges.end());
  for (const auto &edge : graphEdges) {
    if (edge.first >= edge.second) {
      SK_LOGE("ExtractSyncRelationsFromGraph failed: invalid edge index %zu -> %zu", edge.first, edge.second);
      return false;
    }
    InsertSyncEvent(tasks, context, edge.first, edge.second);
  }
  SK_LOGI("ExtractSyncRelationsFromGraph complete: edgeCount=%zu", graphEdges.size());
  return true;
}

bool SkTaskBuilder::ValidateGraphRelations(const std::vector<SkTaskDesc> &tasks, const SkTaskContext &context,
                                           bool allowExternalDependencies) const {
  if (context.nodeIdToIndex.size() != tasks.size() || context.indexToNodeId.size() != tasks.size()) {
    SK_LOGE("ValidateGraphRelations failed: task index is not built for taskCount=%zu", tasks.size());
    return false;
  }
  for (size_t sourceIndex = 0; sourceIndex < tasks.size(); ++sourceIndex) {
    for (uint64_t successorId : tasks[sourceIndex].successorIds) {
      const auto successor = context.nodeIdToIndex.find(successorId);
      if (successor == context.nodeIdToIndex.end()) {
        if (!allowExternalDependencies) {
          SK_LOGE("ValidateGraphRelations failed: missing target nodeId=%lu from source nodeId=%lu", successorId,
                  tasks[sourceIndex].nodeId);
          return false;
        }
        continue;
      }
      if (sourceIndex == successor->second) {
        SK_LOGE("ValidateGraphRelations failed: self dependency nodeId=%lu", tasks[sourceIndex].nodeId);
        return false;
      }
      if (sourceIndex > successor->second) {
        SK_LOGE("ValidateGraphRelations failed: non-topological edge %lu -> %lu (index %zu -> %zu)",
                tasks[sourceIndex].nodeId, successorId, sourceIndex, successor->second);
        return false;
      }
    }
  }
  return true;
}

bool SkTaskBuilder::PrecomputeSyncRelationsFromGraph(const std::vector<SkTaskDesc> &tasks, const SkTaskConfig &config,
                                                     SkTaskContext &context, bool allowExternalDependencies) const {
  (void)config;
  SK_LOGI("Precomputing sync relations from graph, taskCount=%zu", tasks.size());
  if (!InitTaskSyncInfos(tasks, context)) {
    SK_LOGE("PrecomputeSyncRelationsFromGraph failed: InitTaskSyncInfos failed");
    return false;
  }
  if (!ValidateGraphRelations(tasks, context, allowExternalDependencies) ||
      !ExtractSyncRelationsFromGraph(tasks, context, allowExternalDependencies)) {
    SK_LOGE("PrecomputeSyncRelationsFromGraph failed: invalid graph task relations");
    return false;
  }
  return true;
}

void SkTaskBuilder::RemoveSyncInfo(SkTaskContext &context, size_t sendIdx, size_t recvIdx, bool isRemoveRecv,
                                   SyncDirection dirToRemove) const {
  // Equivalent to Python remove_info_by_name.
  auto removeDirection = [](std::map<size_t, SyncDirection> &info, size_t key, SyncDirection toRemove) {
    auto it = info.find(key);
    if (it != info.end() && it->second == toRemove) {
      info.erase(it);
    }
  };

  if (isRemoveRecv) {
    removeDirection(context.syncInfos[sendIdx].vecRecvInfo, recvIdx, dirToRemove);
    removeDirection(context.syncInfos[sendIdx].cubRecvInfo, recvIdx, dirToRemove);
  } else {
    removeDirection(context.syncInfos[sendIdx].vecSendInfo, recvIdx, dirToRemove);
    removeDirection(context.syncInfos[sendIdx].cubSendInfo, recvIdx, dirToRemove);
  }
}

bool SkTaskBuilder::JudgeRemoveCrossSync(const SkTaskContext &context, size_t sendIdx, size_t recvIdx,
                                         bool isCubToVec) const {
  // Equivalent to Python judge_remove.
  // Check whether there is a crossed sync edge.
  // Crossing condition:
  // another edge (send_idx1 -> recv_idx1) exists where recv_idx1 < recv_idx and send_idx1 > send_idx.

  if (isCubToVec) {
    // cub:vec direction
    for (size_t otherRecvIdx = 0; otherRecvIdx < recvIdx; otherRecvIdx++) {
      // Inspect receiver metadata.
      for (auto &kv : context.syncInfos[otherRecvIdx].vecRecvInfo) {
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
      for (auto &kv : context.syncInfos[otherRecvIdx].cubRecvInfo) {
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

void SkTaskBuilder::RemoveCrossedLineSync(SkTaskContext &context) const {
  SK_LOGI("RemoveCrossedLineSync: checking crossed sync relations");

  // Check crossed sync for cub queue (CUB_TO_VEC direction).
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];

    std::vector<size_t> toRemove;
    for (auto &kv : info.cubSendInfo) {
      if (kv.second == SyncDirection::CUB_TO_VEC && JudgeRemoveCrossSync(context, i, kv.first, true)) {
        toRemove.push_back(kv.first);
      }
    }
    for (size_t recvIdx : toRemove) {
      SK_LOGI("  Remove crossed cub:vec sync: task[%zu] -> task[%zu]", i, recvIdx);
      RemoveSyncInfo(context, i, recvIdx, false, SyncDirection::CUB_TO_VEC);
      RemoveSyncInfo(context, recvIdx, i, true, SyncDirection::CUB_TO_VEC);
    }
  }

  // Check crossed sync for vec queue (VEC_TO_CUB direction).
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];

    std::vector<size_t> toRemove;
    for (auto &kv : info.vecSendInfo) {
      if (kv.second == SyncDirection::VEC_TO_CUB && JudgeRemoveCrossSync(context, i, kv.first, false)) {
        toRemove.push_back(kv.first);
      }
    }
    for (size_t recvIdx : toRemove) {
      SK_LOGI("  Remove crossed vec:cub sync: task[%zu] -> task[%zu]", i, recvIdx);
      RemoveSyncInfo(context, i, recvIdx, false, SyncDirection::VEC_TO_CUB);
      RemoveSyncInfo(context, recvIdx, i, true, SyncDirection::VEC_TO_CUB);
    }
  }
}

void SkTaskBuilder::RemoveMultiSendSync(SkTaskContext &context) const {
  SK_LOGI("RemoveMultiSendSync: removing redundant send sync");

  // Check vec queue (VEC_TO_CUB direction).
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];
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
          RemoveSyncInfo(context, i, recvIdx, false, SyncDirection::VEC_TO_CUB);
          RemoveSyncInfo(context, recvIdx, i, true, SyncDirection::VEC_TO_CUB);
        }
      }
    }
  }

  // Check cub queue (CUB_TO_VEC direction).
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];
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
          RemoveSyncInfo(context, i, recvIdx, false, SyncDirection::CUB_TO_VEC);
          RemoveSyncInfo(context, recvIdx, i, true, SyncDirection::CUB_TO_VEC);
        }
      }
    }
  }
}

void SkTaskBuilder::RemoveMultiRecvSync(SkTaskContext &context) const {
  SK_LOGI("RemoveMultiRecvSync: removing redundant recv sync");

  // Check vec queue (CUB_TO_VEC direction).
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];
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
          RemoveSyncInfo(context, i, sendIdx, true, SyncDirection::CUB_TO_VEC);
          RemoveSyncInfo(context, sendIdx, i, false, SyncDirection::CUB_TO_VEC);
        }
      }
    }
  }

  // Check cub queue (VEC_TO_CUB direction).
  for (size_t i = 0; i < context.syncInfos.size(); i++) {
    auto &info = context.syncInfos[i];
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
          RemoveSyncInfo(context, i, sendIdx, true, SyncDirection::VEC_TO_CUB);
          RemoveSyncInfo(context, sendIdx, i, false, SyncDirection::VEC_TO_CUB);
        }
      }
    }
  }
}

void SkTaskBuilder::RemoveRedundantCrossSync(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context) const {
  const size_t invalid = std::numeric_limits<size_t>::max();
  size_t lastAic = invalid;
  size_t lastAiv = invalid;
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (tasks[i].kind == SkTaskKind::KERNEL) {
      if (UsesAic(tasks[i].queueType)) {
        lastAic = i;
      }
      if (UsesAiv(tasks[i].queueType)) {
        lastAiv = i;
      }
      continue;
    }
    if (tasks[i].kind != SkTaskKind::WAIT) {
      continue;
    }
    if (tasks[i].queueType != SkQueueType::AIC && tasks[i].queueType != SkQueueType::AIV) {
      SK_LOGE("Unexpected queue type %s for WAIT node, cannot determine candidate kernel for redundant sync removal.",
              to_string(tasks[i].queueType));
      continue;
    }
    const size_t candidate = tasks[i].queueType == SkQueueType::AIC ? lastAic : lastAiv;
    if (candidate == invalid) {
      continue;
    }
    if (tasks[i].queueType == SkQueueType::AIC && context.syncInfos[candidate].cubSendInfo.empty()) {
      context.syncInfos[candidate].crossSyncInfo.erase(static_cast<size_t>(SkQueueType::AIC));
      SK_LOGI("Remove redundant cross sync: task[%zu] -> task[%zu], remove CUB_TO_CUB hint", candidate, i);
    } else if (tasks[i].queueType == SkQueueType::AIV && context.syncInfos[candidate].vecSendInfo.empty()) {
      context.syncInfos[candidate].crossSyncInfo.erase(static_cast<size_t>(SkQueueType::AIV));
      SK_LOGI("Remove redundant cross sync: task[%zu] -> task[%zu], remove VEC_TO_VEC hint", candidate, i);
    } else {
      SK_LOGI("No redundant cross sync to remove for task[%zu] -> task[%zu]", candidate, i);
    }
  }
}

void SkTaskBuilder::OptimizeSyncRelations(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context) const {
  SK_LOGI("Optimizing Sync Relations");
  // PrintSyncInfo("[INIT STATE]", tasks, context);

  RemoveCrossedLineSync(context);
  // PrintSyncInfo("[AFTER REMOVE CROSSED LINE SYNC]", tasks, context);

  RemoveMultiSendSync(context);
  RemoveMultiRecvSync(context);
  // PrintSyncInfo("[AFTER REMOVE MULTI EVENT SYNC]", tasks, context);

  RemoveRedundantCrossSync(tasks, context);
}

bool SkTaskBuilder::IsMixKernelTask(const SkTaskDesc &task) const {
  const auto *kernel = GetKernel(task);
  return task.kind == SkTaskKind::KERNEL && kernel != nullptr &&
         GetBoolAttribute(kernel->attributes, SkTaskAttributeId::KERNEL_MIX_SPLIT, false);
}

bool SkTaskBuilder::RebaseTaskSyncInfo(TaskSyncInfo &info, size_t offset) const {
  auto rebase = [offset](std::map<size_t, SyncDirection> &values) {
    std::map<size_t, SyncDirection> result;
    for (const auto &value : values) {
      if (value.first > std::numeric_limits<size_t>::max() - offset) {
        SK_LOGE("RebaseTaskSyncInfo failed: sync index overflow, index=%zu, offset=%zu", value.first, offset);
        return false;
      }
      result[value.first + offset] = value.second;
    }
    values.swap(result);
    return true;
  };
  return rebase(info.cubSendInfo) && rebase(info.cubRecvInfo) && rebase(info.vecSendInfo) && rebase(info.vecRecvInfo);
}

bool SkTaskBuilder::PrecomputeSyncRelationsByMixGroups(const std::vector<SkTaskDesc> &tasks, const SkTaskConfig &config,
                                                       SkTaskContext &context, bool allowExternalDependencies) const {
  if (tasks.empty()) {
    SK_LOGE("PrecomputeSyncRelationsByMixGroups failed: empty tasks");
    return false;
  }
  SkTaskContext initial;
  if (!InitTaskSyncInfos(tasks, initial)) {
    SK_LOGE("InitSyncInfoSnapshotForMixGroups failed: InitTaskSyncInfos failed");
    return false;
  }
  if (!ValidateGraphRelations(tasks, initial, allowExternalDependencies)) {
    SK_LOGE("PrecomputeSyncRelationsByMixGroups failed: initial graph validation failed");
    return false;
  }
  if (initial.syncInfos.size() != tasks.size()) {
    SK_LOGE("InitSyncInfoSnapshotForMixGroups failed: init sync info size mismatch, origin=%zu, expected=%zu",
            initial.syncInfos.size(), tasks.size());
    return false;
  }
  size_t splitGroupCount = 1;
  bool hasMixKernel = IsMixKernelTask(tasks.front());
  for (size_t i = 1; i < tasks.size(); ++i) {
    const bool isMix = IsMixKernelTask(tasks[i]);
    hasMixKernel = hasMixKernel || isMix;
    if (isMix != IsMixKernelTask(tasks[i - 1])) {
      ++splitGroupCount;
    }
  }
  SK_LOGI("PrecomputeSyncRelationsByMixGroups: taskCount=%zu, splitGroupCount=%zu, hasMixKernel=%d", tasks.size(),
          splitGroupCount, static_cast<int>(hasMixKernel));
  std::vector<TaskSyncInfo> merged;
  merged.reserve(tasks.size());
  size_t begin = 0;
  size_t groupIndex = 0;
  while (begin < tasks.size()) {
    size_t end = begin + 1;
    while (end < tasks.size() && IsMixKernelTask(tasks[end]) == IsMixKernelTask(tasks[begin])) {
      ++end;
    }
    std::vector<SkTaskDesc> group(tasks.begin() + begin, tasks.begin() + end);
    SK_LOGI(
        "ProcessSyncRelationSplitGroup: process group index=%zu, offset=%zu, size=%zu, isMixGroup=%d, "
        "firstNodeId=%lu, lastNodeId=%lu",
        groupIndex, begin, group.size(), static_cast<int>(IsMixKernelTask(group.front())), group.front().nodeId,
        group.back().nodeId);
    SkTaskContext groupContext;
    groupContext.syncInfos.assign(initial.syncInfos.begin() + begin, initial.syncInfos.begin() + end);
    for (size_t i = 0; i < group.size(); ++i) {
      groupContext.nodeIdToIndex[group[i].nodeId] = i;
      groupContext.indexToNodeId.push_back(group[i].nodeId);
    }
    SK_LOGI("Build sub-phase-1 begin: precompute sync relations, taskCount=%zu, index=%zu", group.size(), groupIndex);
    if (!ValidateGraphRelations(group, groupContext, allowExternalDependencies) ||
        !ExtractSyncRelationsFromGraph(group, groupContext, allowExternalDependencies)) {
      SK_LOGE("ProcessSyncRelationSplitGroup failed: invalid graph task relations");
      return false;
    }
    SK_LOGI("Build sub-phase-2 begin: optimize sync relations, index=%zu", groupIndex);
    OptimizeSyncRelations(group, groupContext);
    if (end < tasks.size()) {
      SK_LOGI("AddBoundaryAllSync: add boundary all-sync at group=%zu, localTask=%zu, globalTask=%zu, nodeId=%lu",
              groupIndex, group.size() - 1, end - 1, group.back().nodeId);
      auto &boundary = groupContext.syncInfos.back();
      boundary.cubSendInfo.clear();
      boundary.vecSendInfo.clear();
      boundary.crossSyncInfo.clear();
      boundary.crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::ALL_SYNC;
    }
    for (auto &info : groupContext.syncInfos) {
      if (!RebaseTaskSyncInfo(info, begin)) {
        SK_LOGE("PrecomputeSyncRelationsByMixGroups failed: sync index rebase overflow, offset=%zu", begin);
        return false;
      }
      merged.push_back(std::move(info));
    }
    SK_LOGI("ProcessSyncRelationSplitGroup: finish group index=%zu, mergedTaskCount=%zu", groupIndex, merged.size());
    begin = end;
    ++groupIndex;
  }
  if (begin != tasks.size() || merged.size() != tasks.size()) {
    SK_LOGE(
        "PrecomputeSyncRelationsByMixGroups failed: merged task sync info size mismatch, consumed=%zu, "
        "merged=%zu, expected=%zu",
        begin, merged.size(), tasks.size());
    return false;
  }
  context.syncInfos = std::move(merged);
  context.nodeIdToIndex.clear();
  context.indexToNodeId.clear();
  for (size_t i = 0; i < tasks.size(); ++i) {
    context.nodeIdToIndex[tasks[i].nodeId] = i;
    context.indexToNodeId.push_back(tasks[i].nodeId);
  }
  SK_LOGI("PrecomputeSyncRelationsByMixGroups complete: taskCount=%zu, splitGroupCount=%zu", tasks.size(),
          splitGroupCount);
  return true;
}

bool SkTaskBuilder::ApplyEarlyStartForQueue(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context,
                                            size_t current, size_t &last, SkQueueType targetQueue) const {
  if (tasks[current].kind != SkTaskKind::KERNEL) {
    last = current;
    return true;
  }
  const auto *kernel = GetKernel(tasks[current]);
  if (kernel == nullptr) {
    return false;
  }
  const bool isAic = targetQueue == SkQueueType::AIC;
  const char *sendInfoName = isAic ? "cubSend" : "vecSend";
  const char *recvInfoName = isAic ? "cubRecv" : "vecRecv";
  const size_t queueId = static_cast<size_t>(targetQueue);
  const SyncDirection selfDirection = isAic ? SyncDirection::CUB_TO_CUB : SyncDirection::VEC_TO_VEC;
  const SkEarlyStartMask selfSet = isAic ? SkEarlyStartMask::AIC_TO_AIC_SET : SkEarlyStartMask::AIV_TO_AIV_SET;
  const SkEarlyStartMask selfWait = isAic ? SkEarlyStartMask::AIC_TO_AIC_WAIT : SkEarlyStartMask::AIV_TO_AIV_WAIT;
  const SkEarlyStartMask crossSet = isAic ? SkEarlyStartMask::AIC_TO_AIV_SET : SkEarlyStartMask::AIV_TO_AIC_SET;
  const SkEarlyStartMask crossWait = isAic ? SkEarlyStartMask::AIV_TO_AIC_WAIT : SkEarlyStartMask::AIC_TO_AIV_WAIT;
  auto &currentSync = context.syncInfos[current];
  if (GetBoolAttribute(kernel->attributes, SkTaskAttributeId::EARLY_START_SET, false)) {
    const auto cross = currentSync.crossSyncInfo.find(queueId);
    if (cross != currentSync.crossSyncInfo.end() && cross->second == selfDirection) {
      currentSync.earlyStartInfo.relatedSetTask = current;
      currentSync.earlyStartInfo.ApplyFuncMask(selfSet);
      currentSync.earlyStartInfo.ApplySyncMask(selfSet);
      currentSync.earlyStartInfo.ApplySyncMask(selfWait);
      currentSync.crossSyncInfo.erase(cross);
      SK_LOGI("early-start rewrite: slot=cur.crossSyncInfo[%s], task[%zu](nodeId=%lu), self sync",
              to_string(targetQueue), current, tasks[current].nodeId);
    }
  }
  const size_t invalid = std::numeric_limits<size_t>::max();
  if (GetBoolAttribute(kernel->attributes, SkTaskAttributeId::EARLY_START_WAIT, false)) {
    auto &currentReceive = isAic ? currentSync.cubRecvInfo : currentSync.vecRecvInfo;
    currentSync.earlyStartInfo.relatedWaitTask = current;
    if (currentReceive.size() > 1) {
      SK_LOGE(
          "ApplyEarlyStartSyncPass failed: slot=cur.%s has multiple syncs, "
          "curTask[%zu](nodeId=%lu), %sSize=%zu",
          recvInfoName, current, tasks[current].nodeId, recvInfoName, currentReceive.size());
      return false;
    }
    if (!currentReceive.empty()) {
      SK_LOGI("early-start absorb: slot=cur.%s, curTask[%zu](nodeId=%lu)", recvInfoName, current,
              tasks[current].nodeId);
      currentSync.earlyStartInfo.ApplySyncMask(crossWait);
      currentSync.earlyStartInfo.ApplyFuncMask(crossWait);
      currentReceive.clear();
    }
    if (last != invalid) {
      auto &previousSync = context.syncInfos[last];
      auto &previousSend = isAic ? previousSync.cubSendInfo : previousSync.vecSendInfo;
      if (previousSend.size() > 1) {
        SK_LOGE(
            "ApplyEarlyStartSyncPass failed: slot=prev.%s has multiple syncs, "
            "prevTask[%zu](nodeId=%lu), curTask[%zu](nodeId=%lu), %sSize=%zu",
            sendInfoName, last, tasks[last].nodeId, current, tasks[current].nodeId, sendInfoName, previousSend.size());
        return false;
      }
      if (isAic) {
        previousSync.earlyStartInfo.nextAicRelatedTask = current;
      } else {
        previousSync.earlyStartInfo.nextAivRelatedTask = current;
      }
      const auto previousCross = previousSync.crossSyncInfo.find(queueId);
      if (previousCross != previousSync.crossSyncInfo.end() && previousCross->second == selfDirection) {
        previousSync.earlyStartInfo.ApplySyncMask(selfSet);
        previousSync.earlyStartInfo.ApplySyncMask(selfWait);
        currentSync.earlyStartInfo.ApplyFuncMask(selfWait);
        previousSync.crossSyncInfo.erase(previousCross);
        SK_LOGI(
            "early-start rewrite: slot=prev.crossSyncInfo[%s], prevTask[%zu](nodeId=%lu) -> "
            "curTask[%zu](nodeId=%lu), self sync",
            to_string(targetQueue), last, tasks[last].nodeId, current, tasks[current].nodeId);
      } else if (previousSync.earlyStartInfo.CheckSyncMask(selfSet)) {
        currentSync.earlyStartInfo.ApplyFuncMask(selfWait);
        SK_LOGI("early-start attach: slot=prev.earlyStartInfo.syncConfig, curTask[%zu](nodeId=%lu), %s wait", current,
                tasks[current].nodeId, to_string(targetQueue));
      }
      if (!previousSend.empty()) {
        SK_LOGI("early-start absorb: slot=prev.%s, prevTask[%zu](nodeId=%lu), curTask[%zu](nodeId=%lu)", sendInfoName,
                last, tasks[last].nodeId, current, tasks[current].nodeId);
        previousSync.earlyStartInfo.ApplySyncMask(crossSet);
        currentSync.earlyStartInfo.ApplyFuncMask(crossSet);
        previousSend.clear();
      }
    }
  }
  last = current;
  return true;
}

bool SkTaskBuilder::ApplyEarlyStartSyncPass(const std::vector<SkTaskDesc> &tasks, const SkTaskConfig &config,
                                            SkTaskContext &context) const {
  SK_LOGI("ApplyEarlyStartSyncPass begin: taskCount=%zu", tasks.size());
  const size_t invalid = std::numeric_limits<size_t>::max();
  size_t lastAic = invalid;
  size_t lastAiv = invalid;
  for (size_t i = 0; i < tasks.size(); ++i) {
    if (UsesAic(tasks[i].queueType) && !ApplyEarlyStartForQueue(tasks, context, i, lastAic, SkQueueType::AIC)) {
      return false;
    }
    if (UsesAiv(tasks[i].queueType) && !ApplyEarlyStartForQueue(tasks, context, i, lastAiv, SkQueueType::AIV)) {
      return false;
    }
    if (config.kernelArch == SkKernelArch::DAV_3510 && tasks[i].kind == SkTaskKind::KERNEL) {
      context.syncInfos[i].earlyStartInfo.ApplyFuncMask(SkEarlyStartMask::SPLIT_CORE_CTRL);
      SK_LOGI("early-start control: nodeId=%lu, set split-core ctrl mask", tasks[i].nodeId);
    }
    if (HasAllSyncBarrier(context.syncInfos[i])) {
      SK_LOGI("early-start barrier: task[%zu](nodeId=%lu) has ALL_SYNC barrier.", i, tasks[i].nodeId);
      lastAic = i;
      lastAiv = i;
    }
  }
  const auto getNodeId = [&tasks](const std::optional<size_t> &index) -> uint64_t {
    return index.has_value() && *index < tasks.size() ? tasks[*index].nodeId : INVALID_TASK_ID;
  };
  for (size_t i = 0; i < context.syncInfos.size(); ++i) {
    const auto &early = context.syncInfos[i].earlyStartInfo;
    SK_LOGI(
        "early-start final: task[%zu](nodeId=%lu), funcConfig=0x%x, syncConfig=0x%x, "
        "relatedSetNodeId=%lu, relatedWaitNodeId=%lu, nextAicNodeId=%lu, nextAivNodeId=%lu",
        i, tasks[i].nodeId, early.funcConfig, early.syncConfig, getNodeId(early.relatedSetTask),
        getNodeId(early.relatedWaitTask), getNodeId(early.nextAicRelatedTask), getNodeId(early.nextAivRelatedTask));
  }
  return true;
}

bool SkTaskBuilder::AddSyncTask(SkTask &task, const SkTaskConfig &config, size_t taskIndex, SkCoreSyncType syncType,
                                uint8_t earlyStartConfig, uint32_t activeCoreCount, SkKernelType relatedType) const {
  TaskQue *queue = task.GetTaskQue();
  if (queue == nullptr) {
    SK_LOGE("AddSyncTask failed: get writable task queue failed");
    return false;
  }
  if (activeCoreCount > std::numeric_limits<uint8_t>::max() || taskIndex > std::numeric_limits<uint32_t>::max()) {
    SK_LOGE("AddSyncTask failed: task index or active core count overflow, taskIndex=%zu, activeCoreCount=%u",
            taskIndex, activeCoreCount);
    return false;
  }
  TaskInfo &info = queue->taskInfos[queue->taskCnt];
  info.index = static_cast<uint32_t>(taskIndex);
  info.type = SkTaskType::TYPE_SYNC;
  info.args = static_cast<uint32_t>(syncType);
  info.numBlocks = static_cast<uint8_t>(activeCoreCount);
  info.extraInfo = earlyStartConfig;
  info.relatedType = relatedType;
  if (GetBoolAttribute(config.attributes, SkTaskAttributeId::CONFIG_DEBUG_ENABLED, false) &&
      GetBoolAttribute(config.attributes, SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL, false)) {
    info.debugOptions |= kDebugOptionDebugSyncAll;
  }
  ++queue->taskCnt;
  return true;
}

bool SkTaskBuilder::DispatchSyncTasks(SkTask &skTaskCube, SkTask &skTaskVec, const SkTaskIR &input,
                                      const SkTaskContext &context, size_t nodeIndex,
                                      const std::map<size_t, SyncDirection> &syncInfo, bool isSend) const {
  const SkQueueType queueType = input.tasks[nodeIndex].queueType;
  for (const auto &kv : syncInfo) {
    size_t peerIdx = kv.first;
    SyncDirection dir = kv.second;
    if (peerIdx >= input.tasks.size()) {
      SK_LOGE("DispatchSyncTasks failed: peer index out of range, taskIndex=%zu, peerIndex=%zu, taskCount=%zu",
              nodeIndex, peerIdx, input.tasks.size());
      return false;
    }
    SkQueueType peerType = input.tasks[peerIdx].queueType;
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
      if (!AddSyncTask(skTaskCube, input.config, nodeIndex, syncType)) {
        SK_LOGE("DispatchSyncTasks failed: add sync task to aic queue failed");
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [flag=%s], [prev=%s, next=%s]", nodeIndex,
              to_string(SkTaskType::TYPE_SYNC), to_string(syncType), to_string(prevType), to_string(nextType));
    }
    if (addToAivQue) {
      if (!AddSyncTask(skTaskVec, input.config, nodeIndex, syncType)) {
        SK_LOGE("DispatchSyncTasks failed: add sync task to aiv queue failed");
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [flag=%s], [prev=%s, next=%s]", nodeIndex,
              to_string(SkTaskType::TYPE_SYNC), to_string(syncType), to_string(prevType), to_string(nextType));
    }
  }
  return true;
}

bool SkTaskBuilder::DispatchSyncTasks(SkTask &aicTask, SkTask &aivTask, const SkTaskIR &input,
                                      const EarlyStartInfo &early, size_t taskIndex, bool send) const {
  const std::map<SkEarlyStartMask, SkCoreSyncType> maskTypes = {
      {SkEarlyStartMask::AIC_TO_AIC_SET, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC},
      {SkEarlyStartMask::AIC_TO_AIC_WAIT, SkCoreSyncType::CROSS_SYNC_AIC_TO_AIC},
      {SkEarlyStartMask::AIV_TO_AIV_SET, SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV},
      {SkEarlyStartMask::AIV_TO_AIV_WAIT, SkCoreSyncType::CROSS_SYNC_AIV_TO_AIV},
      {SkEarlyStartMask::AIC_TO_AIV_SET, SkCoreSyncType::INTER_SYNC_SET_AIC_TO_AIV},
      {SkEarlyStartMask::AIC_TO_AIV_WAIT, SkCoreSyncType::INTER_SYNC_WAIT_AIC_TO_AIV},
      {SkEarlyStartMask::AIV_TO_AIC_SET, SkCoreSyncType::INTER_SYNC_SET_AIV_TO_AIC},
      {SkEarlyStartMask::AIV_TO_AIC_WAIT, SkCoreSyncType::INTER_SYNC_WAIT_AIV_TO_AIC},
  };
  auto addEarlyStartSync = [&](SkTask &target, SkEarlyStartMask mask, const std::optional<size_t> &related,
                               SkQueueType queue) {
    if (!early.CheckSyncMask(mask)) {
      return true;
    }
    const auto syncType = maskTypes.find(mask);
    if (syncType == maskTypes.end()) {
      SK_LOGE("DispatchSyncTasks failed: unknown early start sync mask %s", to_string(mask));
      return false;
    }
    const SkKernelDesc *kernel = nullptr;
    if (related.has_value() && *related < input.tasks.size()) {
      kernel = GetKernel(input.tasks[*related]);
    }
    const uint32_t activeCoreCount = kernel == nullptr ? 0 : GetTaskNumBlocks(*kernel, queue, input.coreSpec);
    const SkKernelType relatedType = kernel == nullptr ? SkKernelType::DEFAULT : kernel->kernelType;
    if (!AddSyncTask(target, input.config, taskIndex, syncType->second, static_cast<uint8_t>(mask), activeCoreCount,
                     relatedType)) {
      SK_LOGE("DispatchSyncTasks failed: add early start sync task failed, mask=%s", to_string(mask));
      return false;
    }
    SK_LOGI("    task insert: task %zu, [queue=%s], [type=%s], [flag=%s], [activeCoreCount=%u]", taskIndex,
            to_string(queue), to_string(SkTaskType::TYPE_SYNC), to_string(syncType->second), activeCoreCount);
    return true;
  };
  if (!send) {
    return addEarlyStartSync(aicTask, SkEarlyStartMask::AIV_TO_AIC_WAIT, early.relatedWaitTask, SkQueueType::AIC) &&
           addEarlyStartSync(aivTask, SkEarlyStartMask::AIC_TO_AIV_WAIT, early.relatedWaitTask, SkQueueType::AIV);
  }
  return addEarlyStartSync(aicTask, SkEarlyStartMask::AIC_TO_AIC_SET, early.relatedSetTask, SkQueueType::AIC) &&
         addEarlyStartSync(aicTask, SkEarlyStartMask::AIC_TO_AIC_WAIT, early.nextAicRelatedTask, SkQueueType::AIC) &&
         addEarlyStartSync(aicTask, SkEarlyStartMask::AIC_TO_AIV_SET, early.nextAicRelatedTask, SkQueueType::AIC) &&
         addEarlyStartSync(aivTask, SkEarlyStartMask::AIV_TO_AIV_SET, early.relatedSetTask, SkQueueType::AIV) &&
         addEarlyStartSync(aivTask, SkEarlyStartMask::AIV_TO_AIV_WAIT, early.nextAivRelatedTask, SkQueueType::AIV) &&
         addEarlyStartSync(aivTask, SkEarlyStartMask::AIV_TO_AIC_SET, early.nextAivRelatedTask, SkQueueType::AIV);
}

bool SkTaskBuilder::AddEventTask(SkTask &task, const SkEventDesc &event, size_t taskIndex, SkTaskType type) const {
  TaskQue *queue = task.GetTaskQue();
  if (queue == nullptr) {
    SK_LOGE("AddEventTask failed: get writable task queue failed");
    return false;
  }
  if (taskIndex > std::numeric_limits<uint32_t>::max()) {
    SK_LOGE("AddEventTask failed: task index overflow, taskIndex=%zu", taskIndex);
    return false;
  }
  TaskInfo &info = queue->taskInfos[queue->taskCnt];
  info.index = static_cast<uint32_t>(taskIndex);
  info.type = type;
  uint64_t value = event.value;
  if (value == std::numeric_limits<uint64_t>::max()) {
    value = type == SkTaskType::TYPE_EVENT_NOTIFY
                ? SK_DEFAULT_NOTIFY_VALUE
                : (type == SkTaskType::TYPE_EVENT_WAIT ? SK_DEFAULT_WAIT_VALUE : SK_DEFAULT_RESET_VALUE);
  }
  uint32_t flag = event.waitFlag;
  if (type != SkTaskType::TYPE_EVENT_WAIT || flag == std::numeric_limits<uint32_t>::max()) {
    flag = type == SkTaskType::TYPE_EVENT_WAIT ? static_cast<uint32_t>(SkMemoryWaitFlag::EQ) : SK_DEFAULT_WRITE_FLAG;
  }
  SetEventTaskArgs(info, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(event.address)), value, flag);
  ++queue->taskCnt;
  return true;
}

bool SkTaskBuilder::DispatchEventTask(SkTask &aicTask, SkTask &aivTask, const SkTaskDesc &task, size_t taskIndex,
                                      SkTaskType type) const {
  const auto *event = GetEvent(task);
  if (event == nullptr || (task.queueType != SkQueueType::AIC && task.queueType != SkQueueType::AIV)) {
    SK_LOGE("DispatchEventTask failed: unsupported node type for EVENT_NOTIFY/EVENT_WAIT/EVENT_RESET task");
    return false;
  }
  if (!AddEventTask(task.queueType == SkQueueType::AIC ? aicTask : aivTask, *event, taskIndex, type)) {
    SK_LOGE("DispatchEventTask failed: add event task failed");
    return false;
  }
  SK_LOGI("    task insert: task %zu, [queue=%s], [type=%s], [nodeInfo=%s]", taskIndex,
          task.queueType == SkQueueType::AIC ? "AIC" : "AIV", to_string(type), task.debugLabel.c_str());
  return true;
}

bool SkTaskBuilder::ProcessCoreFuncSize(SkDfxInfo *dfxInfo, aclrtBinHandle binHdl, const void *binHostAddr,
                                        uint32_t binHostSize, const SkResolvedFunctionDesc &resolved, int coreIndex,
                                        int binIndex, const char *coreName) const {
  SK_LOGD("ProcessCoreFuncSize: Processing %s (coreIndex=%d), funcAddr=0x%lx, funcOffset=0x%lx", coreName, coreIndex,
          resolved.functionAddress[coreIndex], resolved.functionOffset[coreIndex]);

  std::string symbolName;
  uint64_t funcSize = 0;
  std::string symbolBind;
  bool getInfoRet = GetFuncSymbolInfo(binHdl, static_cast<const char *>(binHostAddr), binHostSize,
                                      resolved.functionOffset[coreIndex], symbolName, funcSize, symbolBind);
  SK_LOGD("ProcessCoreFuncSize: GetFuncSymbolInfo(%s) returned=%d, offset=0x%lx, symbolName=%s, size=0x%lx, bind=%s",
          coreName, getInfoRet, resolved.functionOffset[coreIndex], symbolName.c_str(), funcSize, symbolBind.c_str());

  if (getInfoRet) {
    if (coreIndex == 0) {
      dfxInfo->aicSize = static_cast<uint32_t>(funcSize);
      dfxInfo->entryAic[binIndex] = resolved.functionAddress[0];
      dfxInfo->aicFuncOffset[binIndex] = resolved.functionOffset[0];
      SK_LOGD("ProcessCoreFuncSize: Set aicSize=0x%x, entryAic[%d]=0x%lx, aicFuncOffset[%d]=0x%lx", dfxInfo->aicSize,
              binIndex, dfxInfo->entryAic[binIndex], binIndex, dfxInfo->aicFuncOffset[binIndex]);
    } else if (coreIndex == 1) {
      dfxInfo->aivSize = static_cast<uint32_t>(funcSize);
      dfxInfo->entryAiv[binIndex] = resolved.functionAddress[1];
      dfxInfo->aivFuncOffset[binIndex] = resolved.functionOffset[1];
      SK_LOGD("ProcessCoreFuncSize: Set aivSize=0x%x, entryAiv[%d]=0x%lx, aivFuncOffset[%d]=0x%lx", dfxInfo->aivSize,
              binIndex, dfxInfo->entryAiv[binIndex], binIndex, dfxInfo->aivFuncOffset[binIndex]);
    }
    return true;
  }

  SK_LOGW("ProcessCoreFuncSize: GetFuncSymbolInfo failed for %s", coreName);
  return false;
}

bool SkTaskBuilder::UpdateDfxInfo(SkDfxInfo *dfxInfo, const SkKernelDesc &kernelInfo,
                                  const SkResolvedFunctionDesc &resolved, int binIndex, int addrIndex) const {
  if (dfxInfo == nullptr) {
    return true;
  }

  SK_LOGD("UpdateDfxInfo: Processing dfxInfo for binIndex=%d, addrIndex=%d, funcName=%s", binIndex, addrIndex,
          kernelInfo.functionName.c_str());

  dfxInfo->binHdl = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernelInfo.binaryHandle));
  dfxInfo->funcHdlOri = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernelInfo.originalFunction));
  dfxInfo->numBlocks = kernelInfo.originalNumBlocks;
  dfxInfo->cubeNum = kernelInfo.cubeNum;
  dfxInfo->vecNum = kernelInfo.vectorNum;
  SK_LOGD("UpdateDfxInfo: binHdl=0x%lx, funcHdlOri=0x%lx, numBlocks=%u, cubeNum=%u, vecNum=%u", dfxInfo->binHdl,
          dfxInfo->funcHdlOri, dfxInfo->numBlocks, dfxInfo->cubeNum, dfxInfo->vecNum);

  void *binHostAddr = nullptr;
  uint32_t binHostSize = 0;
  int rtRet = rtGetBinBuffer(kernelInfo.binaryHandle, RT_BIN_HOST_ADDR, &binHostAddr, &binHostSize);
  if (rtRet != 0) {
    SK_LOGW("UpdateDfxInfo: Failed to get bin buffer, rtRet=%d, using default size 0", rtRet);
    return true;
  }

  SK_LOGD("UpdateDfxInfo: Successfully got bin buffer, binHostAddr=0x%lx, binHostSize=%u", (uint64_t)binHostAddr,
          binHostSize);

  if (addrIndex == 0 && resolved.functionAddress[0] != 0) {
    ProcessCoreFuncSize(dfxInfo, kernelInfo.binaryHandle, binHostAddr, binHostSize, resolved, 0, binIndex, "AIC");
  } else {
    SK_LOGD("UpdateDfxInfo: Skipping AIC processing, addrIndex=%d, funcAddr[0]=0x%lx", addrIndex,
            resolved.functionAddress[0]);
  }

  if (addrIndex == 1 && resolved.functionAddress[1] != 0) {
    ProcessCoreFuncSize(dfxInfo, kernelInfo.binaryHandle, binHostAddr, binHostSize, resolved, 1, binIndex, "AIV");
  } else {
    SK_LOGD("UpdateDfxInfo: Skipping AIV processing, addrIndex=%d, funcAddr[1]=0x%lx", addrIndex,
            resolved.functionAddress[1]);
  }

  return true;
}

bool SkTaskBuilder::AddFuncTask(SkTask &task, const SkKernelDesc &kernel, const EarlyStartInfo &early, SkDfxInfo *dfx,
                                size_t taskIndex, int addressIndex, uint32_t binaryCount, SkTaskType type,
                                uint32_t numBlocks) const {
  TaskQue *queue = task.GetTaskQue();
  if (queue == nullptr) {
    SK_LOGE("AddFuncTask failed: get writable task queue failed");
    return false;
  }
  if (taskIndex > std::numeric_limits<uint32_t>::max() || numBlocks > std::numeric_limits<uint8_t>::max()) {
    SK_LOGE("AddFuncTask failed: task index or block count overflow, taskIndex=%zu, numBlocks=%u", taskIndex,
            numBlocks);
    return false;
  }
  TaskInfo &info = queue->taskInfos[queue->taskCnt];
  info.index = static_cast<uint32_t>(taskIndex);
  info.type = type;
  info.relatedType = kernel.kernelType;
  info.isSimtKernel =
      type == SkTaskType::TYPE_FUNC && GetBoolAttribute(kernel.attributes, SkTaskAttributeId::SIMT_ENABLED, false) ? 1
                                                                                                                   : 0;
  info.numBlocks = static_cast<uint8_t>(numBlocks);
  if (kernel.functions.size() != binaryCount) {
    SK_LOGW("Mismatch between resolved functions and split count, kernel=%s, resolved=%zu, configured=%u",
            kernel.functionName.c_str(), kernel.functions.size(), binaryCount);
  }
  if (binaryCount > kernel.functions.size() || binaryCount > kMaxSplitBinCount) {
    SK_LOGE("AddFuncTask failed: invalid split binary count, kernel=%s, resolved=%zu, configured=%u",
            kernel.functionName.c_str(), kernel.functions.size(), binaryCount);
    return false;
  }
  for (uint32_t i = 0; i < binaryCount; ++i) {
    const auto &function = kernel.functions[i];
    if (type == SkTaskType::TYPE_PRELOAD) {
      info.args = function.prefetchCount[addressIndex];
      info.argsSize = kernel.argsSize;
      info.extraInfo = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernel.deviceArgs));
      if (i == 0) {
        SK_LOGI("kernel name: %s, prefetch count: %lu %lu", kernel.functionName.c_str(), function.prefetchCount[0],
                function.prefetchCount[1]);
      }
    }
    const uint64_t address = function.functionAddress[addressIndex];
    if (address == 0) {
      SK_LOGE("AddFuncTask failed: unresolved function address, nodeIndex=%zu, binIndex=%u, addrIndex=%d", taskIndex, i,
              addressIndex);
      return false;
    }
    ++info.entryCnt;
    info.entry[i] = address;
    if (type == SkTaskType::TYPE_FUNC && dfx != nullptr) {
      UpdateDfxInfo(dfx, kernel, function, static_cast<int>(i), addressIndex);
    }
  }
  if (type == SkTaskType::TYPE_FUNC) {
    info.args = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernel.deviceArgs));
    info.extraInfo = early.funcConfig;
    info.argsSize = kernel.argsSize;
    info.debugOptions |=
        GetUint64Attribute(kernel.attributes, SkTaskAttributeId::KERNEL_EFFECTIVE_DEBUG_OPTIONS).value_or(0);
    task.numBlocks = std::max(task.numBlocks, numBlocks);
    ++task.funcCnt;
  }
  ++queue->taskCnt;
  return true;
}

bool SkTaskBuilder::DispatchFuncTask(SkTask &skTaskCube, SkTask &skTaskVec, const SkTaskIR &input,
                                     const SkTaskContext &context, size_t nodeIndex, SkDfxInfo *dfxInfo,
                                     SkTaskType taskType) const {
  const auto &task = input.tasks[nodeIndex];
  const auto *kernel = GetKernel(task);
  const auto queueType = task.queueType;
  const auto binCount = input.config.splitBinCount;
  if (task.kind != SkTaskKind::KERNEL || kernel == nullptr) {
    SK_LOGE("DispatchFuncTask failed: unsupported node type for FUNC/PRELOAD task");
    return false;
  }
  const uint32_t cubeNum = UsesAic(queueType) ? GetTaskNumBlocks(*kernel, SkQueueType::AIC, input.coreSpec) : 0;
  const uint32_t vectorNum = UsesAiv(queueType) ? GetTaskNumBlocks(*kernel, SkQueueType::AIV, input.coreSpec) : 0;

  switch (queueType) {
    case SkQueueType::AIV: {
      if (!AddFuncTask(skTaskVec, *kernel, context.syncInfos[nodeIndex].earlyStartInfo, dfxInfo, nodeIndex, 1, binCount,
                       taskType, vectorNum)) {
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), task.debugLabel.c_str());
      break;
    }
    case SkQueueType::AIC: {
      if (!AddFuncTask(skTaskCube, *kernel, context.syncInfos[nodeIndex].earlyStartInfo, dfxInfo, nodeIndex, 0,
                       binCount, taskType, cubeNum)) {
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), task.debugLabel.c_str());
      break;
    }
    case SkQueueType::MIX_1_1: {
      if (!AddFuncTask(skTaskCube, *kernel, context.syncInfos[nodeIndex].earlyStartInfo, dfxInfo, nodeIndex, 0,
                       binCount, taskType, cubeNum)) {
        return false;
      }
      if (!AddFuncTask(skTaskVec, *kernel, context.syncInfos[nodeIndex].earlyStartInfo, dfxInfo, nodeIndex, 1, binCount,
                       taskType, vectorNum)) {
        return false;
      }
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), task.debugLabel.c_str());
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), task.debugLabel.c_str());
      break;
    }
    case SkQueueType::MIX_1_2: {
      if (!AddFuncTask(skTaskCube, *kernel, context.syncInfos[nodeIndex].earlyStartInfo, dfxInfo, nodeIndex, 0,
                       binCount, taskType, cubeNum)) {
        return false;
      }
      if (!AddFuncTask(skTaskVec, *kernel, context.syncInfos[nodeIndex].earlyStartInfo, dfxInfo, nodeIndex, 1, binCount,
                       taskType, vectorNum)) {
        return false;
      }
      skTaskVec.nodeType = SkKernelType::MIX_AIC_1_2;
      skTaskCube.nodeType = SkKernelType::MIX_AIC_1_2;
      SK_LOGI("    task insert: task %zu, [queue=AIC], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), task.debugLabel.c_str());
      SK_LOGI("    task insert: task %zu, [queue=AIV], [type=%s], [kernelType=%s], [nodeInfo=%s]", nodeIndex,
              to_string(taskType), to_string(queueType), task.debugLabel.c_str());
      break;
    }
    default:
      SK_LOGE("DispatchFuncTask failed: unsupported kernel type %s for super kernel", to_string(queueType));
      return false;
  }
  return true;
}

SkHostEntryInfo SkTaskBuilder::GenEntryInfo(SkTask &skTaskCube, SkTask &skTaskVec, const SkTaskIR &input,
                                            bool useSimtEntry) const {
  SkHostEntryInfo entryInfo;
  bool enableDebug = GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_ENABLED, false);
  bool enableProfiling = GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_PROFILING_ENABLED, false);
  bool enableOpTrace = GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_OP_TRACE_ENABLED, false);
  bool enableEarlyStart =
      GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED, false);

  SK_LOGI(
      "GenEntryInfo: enableDebug=%d, enableProfiling=%d, enableOpTrace=%d, enableEarlyStart=%d, "
      "useSimtEntry=%d",
      enableDebug, enableProfiling, enableOpTrace, enableEarlyStart, useSimtEntry);

  // Scope splitting finalizes the SK core result; TaskBuilder only consumes it.
  entryInfo.entryType = input.coreSpec.type;
  entryInfo.numBlocks = input.coreSpec.numBlocks;
  skTaskCube.numBlocks = input.coreSpec.cubeNum;
  skTaskVec.numBlocks = input.coreSpec.vectorNum;

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

DeviceArgsPtr SkTaskBuilder::GenEntryArgs(const SkTask &aicTask, const SkTask &aivTask, const SkTaskIR &input,
                                          const SkDfxInfo *dfxInfos) const {
  const size_t headerSize = sizeof(SkHeaderInfo);
  const size_t aicQueueSize = aicTask.GetTaskQueSize();
  const size_t aivQueueSize = aivTask.GetTaskQueSize();
  size_t aicQueueCopiesSize = 0;
  size_t aivQueueCopiesSize = 0;
  size_t counterSize = 0;
  size_t dfxSize = 0;
  if (!CheckedMultiply(aicQueueSize, 4U, aicQueueCopiesSize) ||
      !CheckedMultiply(aivQueueSize, 4U, aivQueueCopiesSize) ||
      !CheckedMultiply(input.config.counterSlotCount, sizeof(SkCounterInfo), counterSize) ||
      !CheckedMultiply(input.tasks.size(), sizeof(SkDfxInfo), dfxSize)) {
    SK_LOGE(
        "GenEntryArgs failed: argument size overflow, aicQueSize=%zu, aivQueSize=%zu, "
        "counterSlotCount=%u, taskCount=%zu",
        aicQueueSize, aivQueueSize, input.config.counterSlotCount, input.tasks.size());
    return {};
  }
  const size_t eventConfigSize = sizeof(SkEventConfig);
  const size_t aicQueueOffset = headerSize;
  size_t aivQueueOffset = 0;
  size_t unalignedCounterOffset = 0;
  size_t counterOffset = 0;
  size_t dfxOffset = 0;
  size_t eventConfigOffset = 0;
  size_t totalSize = 0;
  if (!CheckedAdd(aicQueueOffset, aicQueueCopiesSize, aivQueueOffset) ||
      !CheckedAdd(aivQueueOffset, aivQueueCopiesSize, unalignedCounterOffset) ||
      !CheckedAlignUp(unalignedCounterOffset, kCounterAlignBytes, counterOffset) ||
      !CheckedAdd(counterOffset, counterSize, dfxOffset) || !CheckedAdd(dfxOffset, dfxSize, eventConfigOffset) ||
      !CheckedAdd(eventConfigOffset, eventConfigSize, totalSize) ||
      aicQueueSize > std::numeric_limits<uint32_t>::max() || aivQueueSize > std::numeric_limits<uint32_t>::max() ||
      aicQueueOffset > std::numeric_limits<uint32_t>::max() || aivQueueOffset > std::numeric_limits<uint32_t>::max() ||
      counterOffset > std::numeric_limits<uint32_t>::max() || dfxOffset > std::numeric_limits<uint32_t>::max() ||
      eventConfigOffset > std::numeric_limits<uint32_t>::max()) {
    SK_LOGE("GenEntryArgs failed: invalid task argument layout");
    return {};
  }
  SK_LOGI(
      "sk total args size: %zu, header_size: %zu, aic_que_size: %zu, aiv_que_size: %zu, counter_size: %zu, "
      "dfx_size: %zu, counter_offset: %zu",
      totalSize, headerSize, aicQueueSize, aivQueueSize, counterSize, dfxSize, counterOffset);
  DeviceArgsPtr args;
  if (!args.Init(totalSize)) {
    SK_LOGE("GenEntryArgs init device args failed, total_size=%zu", totalSize);
    return {};
  }
  SkHeaderInfo &header = args.Get()->skHeader;
  header.aicQueSize = aicQueueSize;
  header.aivQueSize = aivQueueSize;
  header.aicQueOffset = aicQueueOffset;
  header.aivQueOffset = aivQueueOffset;
  header.counterOffset = counterOffset;
  header.dfxOffset = dfxOffset;
  header.eventConfigOffset = eventConfigOffset;
  header.totalSize = totalSize;
  header.nodeCnt = static_cast<uint32_t>(input.tasks.size());
  uint8_t *base = reinterpret_cast<uint8_t *>(args.Get());
  for (size_t i = 0; i < 4; ++i) {
    const errno_t err =
        memcpy_s(base + aicQueueOffset + i * aicQueueSize, aicQueueSize, aicTask.GetTaskQue(), aicQueueSize);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s AIC queue%zu failed, ret=%d", i + 1, static_cast<int>(err));
      return {};
    }
  }
  for (size_t i = 0; i < 4; ++i) {
    const errno_t err =
        memcpy_s(base + aivQueueOffset + i * aivQueueSize, aivQueueSize, aivTask.GetTaskQue(), aivQueueSize);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s AIV queue%zu failed, ret=%d", i + 1, static_cast<int>(err));
      return {};
    }
  }
  if (counterSize != 0) {
    const errno_t err = memset_s(base + counterOffset, counterSize, 0, counterSize);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memset_s counter failed, ret=%d", static_cast<int>(err));
      return {};
    }
  }
  if (dfxSize != 0) {
    const errno_t err = memcpy_s(base + dfxOffset, dfxSize, dfxInfos, dfxSize);
    if (err != 0) {
      SK_LOGE("GenEntryArgs memcpy_s dfx failed, ret=%d", static_cast<int>(err));
      return {};
    }
  }
  const errno_t eventErr = memset_s(base + eventConfigOffset, eventConfigSize, 0, eventConfigSize);
  if (eventErr != 0) {
    SK_LOGE("GenEntryArgs memset_s eventConfig failed, ret=%d", static_cast<int>(eventErr));
    return {};
  }
  reinterpret_cast<SkEventConfig *>(base + eventConfigOffset)->enabled = 0;
  return args;
}

bool SkTaskBuilder::CalculateSkScheMode(const SkTaskIR &input, const SkTask &aicTask, const SkTask &aivTask) const {
  for (const SkTask *skTask : {&aicTask, &aivTask}) {
    const TaskQue *queue = skTask->GetTaskQue();
    if (queue == nullptr) {
      continue;
    }
    for (uint32_t i = 0; i < queue->taskCnt; ++i) {
      const TaskInfo &taskInfo = queue->taskInfos[i];
      if (taskInfo.type == SkTaskType::TYPE_FUNC) {
        if (taskInfo.index < input.tasks.size()) {
          const auto *kernel = GetKernel(input.tasks[taskInfo.index]);
          if (kernel != nullptr && GetBoolAttribute(kernel->attributes, SkTaskAttributeId::KERNEL_SCHE_MODE, false)) {
            return true;
          }
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

bool SkTaskBuilder::CalculateSimtDcacheSize(const SkTaskIR &input, bool &useSimtEntry, size_t &maxDcacheSize) const {
  useSimtEntry = false;
  maxDcacheSize = 0;
  if (!GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_SIMT_ENABLED, false)) {
    return true;
  }
  maxDcacheSize = SK_TOTAL_UB_SIZE;
  for (const auto &task : input.tasks) {
    const auto *kernel = GetKernel(task);
    if (kernel == nullptr || !GetBoolAttribute(kernel->attributes, SkTaskAttributeId::SIMT_ENABLED, false)) {
      continue;
    }
    useSimtEntry = true;
    const auto dynUbufAttribute = GetUint64Attribute(kernel->attributes, SkTaskAttributeId::SIMT_DYN_UBUF_SIZE);
    const auto allocUbufAttribute = GetUint64Attribute(kernel->attributes, SkTaskAttributeId::SIMT_ALLOC_UBUF_SIZE);
    if (!dynUbufAttribute.has_value() || !allocUbufAttribute.has_value() ||
        dynUbufAttribute.value() > std::numeric_limits<size_t>::max() ||
        allocUbufAttribute.value() > std::numeric_limits<size_t>::max()) {
      SK_LOGE("SIMT kernel lacks ubuf size info, nodeId=%lu, hasDynUbufSize=%d, hasAllocUbufSize=%d", task.nodeId,
              dynUbufAttribute.has_value(), allocUbufAttribute.has_value());
      return false;
    }
    const size_t dynUbufSize = static_cast<size_t>(dynUbufAttribute.value());
    const size_t allocUbufSize = static_cast<size_t>(allocUbufAttribute.value());
    if (dynUbufSize > SK_TOTAL_UB_SIZE || allocUbufSize > SK_TOTAL_UB_SIZE - dynUbufSize) {
      SK_LOGE(
          "SIMT kernel ubuf size exceeds total ub size, nodeId=%lu, dynUbufSize=%zu, "
          "allocUbufSize=%zu, totalUbSize=%zu",
          task.nodeId, dynUbufSize, allocUbufSize, SK_TOTAL_UB_SIZE);
      return false;
    }
    maxDcacheSize = std::min(maxDcacheSize, SK_TOTAL_UB_SIZE - dynUbufSize - allocUbufSize);
  }
  if (!useSimtEntry) {
    maxDcacheSize = 0;
  }
  return true;
}

bool SkTaskBuilder::ValidateInput(const SkTaskIR &input) const {
  if (input.tasks.empty()) {
    SK_LOGE("Build failed: no task to build for super kernel");
    return false;
  }
  if (!input.coreSpec.IsValid()) {
    SK_LOGE("Build failed: scope %u has no final SK core result", input.scopeId);
    return false;
  }
  if (input.tasks.size() > std::numeric_limits<uint32_t>::max() || input.config.splitBinCount == 0 ||
      input.config.splitBinCount > kMaxSplitBinCount || input.config.counterSlotCount == 0 ||
      !ValidateTaskConfigAttributes(input.config.attributes)) {
    SK_LOGE("Build failed: invalid task count or configuration, taskCount=%zu, splitBinCount=%u, counterSlotCount=%u",
            input.tasks.size(), input.config.splitBinCount, input.config.counterSlotCount);
    return false;
  }
  bool hasKernel = false;
  std::unordered_set<uint64_t> nodeIds;
  for (size_t i = 0; i < input.tasks.size(); ++i) {
    const auto &task = input.tasks[i];
    if (!IsValidTaskKind(task.kind)) {
      SK_LOGE("process task %zu: unsupported node type %s, skipping (supported types: KERNEL, NOTIFY, WAIT, RESET)", i,
              to_string(task.kind));
      return false;
    }
    if (task.nodeId == INVALID_TASK_ID) {
      SK_LOGE("BuildTaskIndex failed: invalid node ID at index=%zu", i);
      return false;
    }
    if (!nodeIds.insert(task.nodeId).second) {
      SK_LOGE("BuildTaskIndex failed: duplicate nodeId=%lu, index=%zu", task.nodeId, i);
      return false;
    }
    if (task.queueType == SkQueueType::UNKNOWN) {
      SK_LOGE("queue type was not prepared by optimizer, nodeId=%lu", task.nodeId);
      return false;
    }
    const bool kernelKind = task.kind == SkTaskKind::KERNEL;
    if (kernelKind != (GetKernel(task) != nullptr)) {
      SK_LOGE("Build failed: task payload does not match task kind, nodeId=%lu, kind=%s", task.nodeId,
              to_string(task.kind));
      return false;
    }
    if (kernelKind && task.queueType != ExpectedKernelQueue(GetKernel(task)->kernelType)) {
      SK_LOGE("Build failed: kernel queue type mismatch, nodeId=%lu, kernelType=%s, queueType=%s", task.nodeId,
              to_string(GetKernel(task)->kernelType), to_string(task.queueType));
      return false;
    }
    if (kernelKind) {
      hasKernel = true;
      const auto &kernel = *GetKernel(task);
      const bool doublesVectorBlocks =
          UsesAiv(task.queueType) && input.coreSpec.type == SkKernelType::MIX_AIC_1_2 &&
          (kernel.kernelType == SkKernelType::MIX_AIC_1_1 || kernel.kernelType == SkKernelType::MIX_AIC_1_2);
      if (!ValidateKernelAttributes(kernel.attributes) || kernel.functions.size() != input.config.splitBinCount ||
          (doublesVectorBlocks && kernel.originalNumBlocks > std::numeric_limits<uint32_t>::max() / 2U) ||
          (UsesAic(task.queueType) && input.coreSpec.cubeNum == 0) ||
          (UsesAiv(task.queueType) && input.coreSpec.vectorNum == 0)) {
        SK_LOGE(
            "Build failed: invalid kernel attributes, functions or core requirement, "
            "nodeId=%lu, functionCount=%zu, splitBinCount=%u",
            task.nodeId, kernel.functions.size(), input.config.splitBinCount);
        return false;
      }
    }
    if (!kernelKind &&
        (GetEvent(task) == nullptr || (task.queueType != SkQueueType::AIC && task.queueType != SkQueueType::AIV))) {
      SK_LOGE("Build failed: invalid event payload or queue type, nodeId=%lu, queueType=%s", task.nodeId,
              to_string(task.queueType));
      return false;
    }
  }
  if (!hasKernel) {
    SK_LOGE("Build failed: task input contains no kernel, scopeId=%u", input.scopeId);
  }
  return hasKernel;
}

SkBuildResult SkTaskBuilder::Build(const SkTaskIR &input) const {
  if (!ValidateInput(input)) {
    SK_LOGE("Build failed: invalid lowered task input, scopeId=%u", input.scopeId);
    return {};
  }
  const size_t capacity = input.tasks.size() * static_cast<uint8_t>(SkTaskType::TYPE_MAX);
  SkTask aicTask;
  SkTask aivTask;
  if (!aicTask.Init(capacity)) {
    SK_LOGE("Build failed: init aic task queue failed, cap=%zu", capacity);
    return {};
  }
  if (!aivTask.Init(capacity)) {
    SK_LOGE("Build failed: init aiv task queue failed, cap=%zu", capacity);
    return {};
  }
  if (input.tasks.size() > std::numeric_limits<size_t>::max() / sizeof(SkDfxInfo)) {
    SK_LOGE("invalid dfxInfos alloc size, taskCount=%zu", input.tasks.size());
    return {};
  }
  std::vector<SkDfxInfo> dfxInfos(input.tasks.size());
  const errno_t initErr =
      memset_s(dfxInfos.data(), dfxInfos.size() * sizeof(SkDfxInfo), 0, dfxInfos.size() * sizeof(SkDfxInfo));
  if (initErr != 0) {
    SK_LOGE("init dfxInfos failed, ret=%d", static_cast<int>(initErr));
    return {};
  }

  SkTaskContext context;
  constexpr bool allowExternalDependencies = true;
  if (GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_MIX_KERNEL_SPLIT_ENABLED, false)) {
    if (!PrecomputeSyncRelationsByMixGroups(input.tasks, input.config, context, allowExternalDependencies)) {
      SK_LOGE("Build failed: precompute sync relations with mix kernel split failed");
      return {};
    }
  } else {
    SK_LOGI("Build phase-1 begin: precompute sync relations, taskCount=%zu", input.tasks.size());
    if (!PrecomputeSyncRelationsFromGraph(input.tasks, input.config, context, allowExternalDependencies)) {
      SK_LOGE("Build failed: precompute sync relations failed");
      return {};
    }
    SK_LOGI("Build phase-2 begin: optimize sync relations");
    OptimizeSyncRelations(input.tasks, context);
  }
  const bool earlyStartEnabled =
      GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_EARLY_START_ENABLED, false);
  if (earlyStartEnabled && !ApplyEarlyStartSyncPass(input.tasks, input.config, context)) {
    SK_LOGE("Build failed: apply early-start sync pass failed");
    return {};
  }
  if (GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_ENABLED, false) &&
      GetBoolAttribute(input.config.attributes, SkTaskAttributeId::CONFIG_DEBUG_SYNC_ALL, false)) {
    SK_LOGI("tracing sync all is enabled, now clear all sync info, and only left cross sync info. task count is %zu",
            input.tasks.size());
    for (auto &sync : context.syncInfos) {
      sync = {};
      sync.crossSyncInfo[static_cast<size_t>(SkQueueType::AIC)] = SyncDirection::ALL_SYNC;
    }
  }

  constexpr size_t preloadDistance = 1;
  SK_LOGI("start build tasks for super kernel, taskCount=%zu", input.tasks.size());
  SK_LOGI("add tasks preload, preload count is %zu, task count is %zu", preloadDistance, input.tasks.size());
  for (size_t i = 0; i < preloadDistance && i < input.tasks.size(); ++i) {
    if (input.tasks[i].kind == SkTaskKind::KERNEL) {
      if (!DispatchFuncTask(aicTask, aivTask, input, context, i, &dfxInfos[i], SkTaskType::TYPE_PRELOAD)) {
        SK_LOGE("Build failed: preload dispatch failed at task index %zu", i);
        return {};
      }
    }
  }
  SK_LOGI("start dispatch tasks...");
  for (size_t i = 0; i < input.tasks.size(); ++i) {
    const auto &task = input.tasks[i];
    const auto &sync = context.syncInfos[i];
    SK_LOGI("index=%zu, nodeType=%s, nodeInfo=%s", i, to_string(task.kind), task.debugLabel.c_str());
    if (earlyStartEnabled && !DispatchSyncTasks(aicTask, aivTask, input, sync.earlyStartInfo, i, false)) {
      return {};
    }
    SK_LOGI("add task sync for vec recv, vec recv size %zu", sync.vecRecvInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, input, context, i, sync.vecRecvInfo, false)) {
      SK_LOGE("Build failed: dispatch vec recv sync failed at task index %zu", i);
      return {};
    }
    SK_LOGI("add task sync for cub recv, cub recv size %zu", sync.cubRecvInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, input, context, i, sync.cubRecvInfo, false)) {
      SK_LOGE("Build failed: dispatch cub recv sync failed at task index %zu", i);
      return {};
    }
    if (task.kind == SkTaskKind::KERNEL) {
      SK_LOGI("add task func, task index is %zu", i);
      if (!DispatchFuncTask(aicTask, aivTask, input, context, i, &dfxInfos[i], SkTaskType::TYPE_FUNC)) {
        SK_LOGE("Build failed: function dispatch failed at task index %zu", i);
        return {};
      }
    } else {
      switch (task.kind) {
        case SkTaskKind::NOTIFY:
          SK_LOGI("add task notify, task index is %zu", i);
          if (!DispatchEventTask(aicTask, aivTask, task, i, EventTaskType(task.kind))) {
            SK_LOGE("Build failed: notify dispatch failed at task index %zu", i);
            return {};
          }
          break;
        case SkTaskKind::WAIT:
          SK_LOGI("add task wait, task index is %zu", i);
          if (!DispatchEventTask(aicTask, aivTask, task, i, EventTaskType(task.kind))) {
            SK_LOGE("Build failed: wait dispatch failed at task index %zu", i);
            return {};
          }
          break;
        case SkTaskKind::RESET:
          SK_LOGI("add task reset, task index is %zu", i);
          if (!DispatchEventTask(aicTask, aivTask, task, i, EventTaskType(task.kind))) {
            SK_LOGE("Build failed: reset dispatch failed at task index %zu", i);
            return {};
          }
          break;
        default:
          SK_LOGE("process task %zu: unsupported node type %s, skipping (supported types: KERNEL, NOTIFY, WAIT, RESET)",
                  i, to_string(task.kind));
          return {};
      }
    }
    if (preloadDistance > 0 && i + preloadDistance < input.tasks.size() &&
        input.tasks[i + preloadDistance].kind == SkTaskKind::KERNEL) {
      SK_LOGI("add tasks preload, task index is %zu", i + preloadDistance);
      if (!DispatchFuncTask(aicTask, aivTask, input, context, i + preloadDistance, &dfxInfos[i + preloadDistance],
                            SkTaskType::TYPE_PRELOAD)) {
        SK_LOGE("Build failed: rolling preload dispatch failed at task index %zu", i + preloadDistance);
        return {};
      }
    }
    if (earlyStartEnabled && !DispatchSyncTasks(aicTask, aivTask, input, sync.earlyStartInfo, i, true)) {
      return {};
    }
    SK_LOGI("add sync task for cross sync info, cross sync info size %zu", sync.crossSyncInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, input, context, i, sync.crossSyncInfo, true)) {
      SK_LOGE("Build failed: dispatch cross sync failed at task index %zu", i);
      return {};
    }
    SK_LOGI("add sync task for vec send, vec send size %zu", sync.vecSendInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, input, context, i, sync.vecSendInfo, true)) {
      SK_LOGE("Build failed: dispatch vec send sync failed at task index %zu", i);
      return {};
    }
    SK_LOGI("add sync task for cub send, cub send size %zu", sync.cubSendInfo.size());
    if (!DispatchSyncTasks(aicTask, aivTask, input, context, i, sync.cubSendInfo, true)) {
      SK_LOGE("Build failed: dispatch cub send sync failed at task index %zu", i);
      return {};
    }
  }
  SK_LOGI("Finish build tasks for super kernel: aicTaskCnt=%u, aivTaskCnt=%u", aicTask.GetTaskQue()->taskCnt,
          aivTask.GetTaskQue()->taskCnt);

  bool useSimtEntry = false;
  size_t maxDcacheSize = 0;
  if (!CalculateSimtDcacheSize(input, useSimtEntry, maxDcacheSize)) {
    SK_LOGE("Build failed: SIMT ubuf size validation failed");
    return {};
  }
  SK_LOGI("Get entry info...");
  SkHostEntryInfo entryInfo = GenEntryInfo(aicTask, aivTask, input, useSimtEntry);
  if (entryInfo.skEntryFunc == nullptr) {
    SK_LOGE("Build failed: GenEntryInfo failed");
    return {};
  }
  entryInfo.nodeCnt = static_cast<uint32_t>(input.tasks.size());
  SK_LOGI("Get entry args...");
  DeviceArgsPtr args = GenEntryArgs(aicTask, aivTask, input, dfxInfos.data());
  if (args.Get() == nullptr) {
    SK_LOGE("Build failed: GenEntryArgs failed");
    return {};
  }
  DumpDeviceArgs(input.functionName, args.Get(), input.tasks);

  SkLaunchInfo launchInfo;
  launchInfo.entryInfo = std::move(entryInfo);
  launchInfo.devArgs = std::move(args);
  launchInfo.skFuncName = input.functionName;
  launchInfo.useSimtEntry = useSimtEntry;
  launchInfo.skMaxDcacheSize = maxDcacheSize;
  launchInfo.isScheModeOn = CalculateSkScheMode(input, aicTask, aivTask);
  SkBuildResult result;
  result.launchInfo = std::move(launchInfo);
  SK_LOGI("SkTaskToQueueJson: generating task queue JSON for scopeId=%u", input.scopeId);
  result.taskQueueJson = SkTaskToQueueJson(aicTask, aivTask, input.scopeId);
  return result;
}
