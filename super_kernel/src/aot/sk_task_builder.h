/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __SK_TASK_BUILDER_H__
#define __SK_TASK_BUILDER_H__

#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "sk_task_ir.h"
#include "sk_types.h"

using Json = nlohmann::ordered_json;

enum class SyncDirection : uint8_t {
  NONE = 0,
  CUB_TO_CUB,
  VEC_TO_VEC,
  CUB_TO_VEC,
  VEC_TO_CUB,
  MIX_TO_MIX,
  ALL_SYNC,
};

struct EarlyStartInfo {
  uint32_t funcConfig = 0;
  uint32_t syncConfig = 0;
  std::optional<size_t> relatedSetTask;
  std::optional<size_t> relatedWaitTask;
  std::optional<size_t> nextAicRelatedTask;
  std::optional<size_t> nextAivRelatedTask;

  void ApplyFuncMask(SkEarlyStartMask mask) {
    funcConfig |= static_cast<uint32_t>(mask);
  }
  void ApplySyncMask(SkEarlyStartMask mask) {
    syncConfig |= static_cast<uint32_t>(mask);
  }
  bool CheckSyncMask(SkEarlyStartMask mask) const {
    return (syncConfig & static_cast<uint32_t>(mask)) != 0;
  }
};

struct TaskSyncInfo {
  std::map<size_t, SyncDirection> cubSendInfo;
  std::map<size_t, SyncDirection> cubRecvInfo;
  std::map<size_t, SyncDirection> vecSendInfo;
  std::map<size_t, SyncDirection> vecRecvInfo;
  std::map<size_t, SyncDirection> crossSyncInfo;
  EarlyStartInfo earlyStartInfo;
};

struct SkBuildResult {
  SkLaunchInfo launchInfo;
  Json taskQueueJson;
};

class SkTaskBuilder {
 public:
  SkBuildResult Build(const SkTaskIR &input) const;

 private:
  struct SkTaskContext {
    std::vector<TaskSyncInfo> syncInfos;
    std::unordered_map<uint64_t, size_t> nodeIdToIndex;
    std::vector<uint64_t> indexToNodeId;
  };

  // Synchronization analysis uses a context local to each Build call.
  bool InitTaskSyncInfos(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context) const;
  bool ValidateGraphRelations(const std::vector<SkTaskDesc> &tasks, const SkTaskContext &context,
                              bool allowExternalDependencies) const;
  bool ExtractSyncRelationsFromGraph(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context,
                                     bool allowExternalDependencies) const;
  bool PrecomputeSyncRelationsFromGraph(const std::vector<SkTaskDesc> &tasks, const SkTaskConfig &config,
                                        SkTaskContext &context, bool allowExternalDependencies) const;
  bool PrecomputeSyncRelationsByMixGroups(const std::vector<SkTaskDesc> &tasks, const SkTaskConfig &config,
                                          SkTaskContext &context, bool allowExternalDependencies) const;
  bool IsMixKernelTask(const SkTaskDesc &task) const;
  bool RebaseTaskSyncInfo(TaskSyncInfo &info, size_t offset) const;
  void InsertSyncEvent(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context, size_t previousIndex,
                       size_t currentIndex) const;

  void OptimizeSyncRelations(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context) const;
  void PrintSyncInfo(const char *stage, const std::vector<SkTaskDesc> &tasks, const SkTaskContext &context) const;
  bool JudgeRemoveCrossSync(const SkTaskContext &context, size_t sender, size_t receiver, bool cubToVec) const;
  void RemoveSyncInfo(SkTaskContext &context, size_t owner, size_t peer, bool receive, SyncDirection direction) const;
  void RemoveCrossedLineSync(SkTaskContext &context) const;
  void RemoveMultiSendSync(SkTaskContext &context) const;
  void RemoveMultiRecvSync(SkTaskContext &context) const;
  void RemoveRedundantCrossSync(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context) const;
  bool ApplyEarlyStartForQueue(const std::vector<SkTaskDesc> &tasks, SkTaskContext &context, size_t current,
                               size_t &last, SkQueueType targetQueue) const;
  bool ApplyEarlyStartSyncPass(const std::vector<SkTaskDesc> &tasks, const SkTaskConfig &config,
                               SkTaskContext &context) const;

  // Task insertion and dispatch consume only lowered descriptions.
  bool AddSyncTask(SkTask &task, const SkTaskConfig &config, size_t taskIndex, SkCoreSyncType syncType,
                   uint8_t earlyStartConfig = 0, uint32_t activeCoreCount = 0,
                   SkKernelType relatedType = SkKernelType::DEFAULT) const;
  bool AddEventTask(SkTask &task, const SkEventDesc &event, size_t taskIndex, SkTaskType type) const;
  bool AddFuncTask(SkTask &task, const SkKernelDesc &kernel, const EarlyStartInfo &early, SkDfxInfo *dfx,
                   size_t taskIndex, int addressIndex, uint32_t binaryCount, SkTaskType type, uint32_t numBlocks) const;
  bool DispatchFuncTask(SkTask &aicTask, SkTask &aivTask, const SkTaskIR &input, const SkTaskContext &context,
                        size_t taskIndex, SkDfxInfo *dfx, SkTaskType type) const;
  bool DispatchEventTask(SkTask &aicTask, SkTask &aivTask, const SkTaskDesc &task, size_t taskIndex,
                         SkTaskType type) const;
  bool DispatchSyncTasks(SkTask &aicTask, SkTask &aivTask, const SkTaskIR &input, const SkTaskContext &context,
                         size_t taskIndex, const std::map<size_t, SyncDirection> &relations, bool send) const;
  bool DispatchSyncTasks(SkTask &aicTask, SkTask &aivTask, const SkTaskIR &input, const EarlyStartInfo &early,
                         size_t taskIndex, bool send) const;

  SkHostEntryInfo GenEntryInfo(SkTask &aicTask, SkTask &aivTask, const SkTaskIR &input, bool simtEntry) const;
  DeviceArgsPtr GenEntryArgs(const SkTask &aicTask, const SkTask &aivTask, const SkTaskIR &input,
                             const SkDfxInfo *dfxInfos) const;
  bool CalculateSkScheMode(const SkTaskIR &input, const SkTask &aicTask, const SkTask &aivTask) const;
  bool CalculateSimtDcacheSize(const SkTaskIR &input, bool &useSimtEntry, size_t &maxDcacheSize) const;
  bool ValidateInput(const SkTaskIR &input) const;

  bool UpdateDfxInfo(SkDfxInfo *dfxInfo, const SkKernelDesc &kernelInfo, const SkResolvedFunctionDesc &resolved,
                     int binIndex, int addrIndex) const;
  bool ProcessCoreFuncSize(SkDfxInfo *dfxInfo, aclrtBinHandle binHdl, const void *binHostAddr, uint32_t binHostSize,
                           const SkResolvedFunctionDesc &resolved, int coreIndex, int binIndex,
                           const char *coreName) const;
};

#endif  // __SK_TASK_BUILDER_H__
